import argparse
import os
import PIL.Image
import torch
import numpy as np
from janus.janusflow.models import MultiModalityCausalLM, VLChatProcessor
import torchvision
from transformers.cache_utils import DynamicCache
import random

from modeling.janusflow.janusflow import add_argus_arguments, init_attn_mlp_from_args


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Generate images using Bagel model.")
	parser = add_argus_arguments(parser)
	args = parser.parse_args()
	base_attention, mlp_args = init_attn_mlp_from_args(args, 30, 24)

	# specify the path to the model
	model_path = "./models/JanusFlow-1.3B"
	vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
	tokenizer = vl_chat_processor.tokenizer

	vl_gpt = MultiModalityCausalLM.from_pretrained(
			model_path, trust_remote_code=True, base_attn=base_attention, mlp_args=mlp_args
	)
	vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

	from diffusers.models import AutoencoderKL
	# remember to use bfloat16 dtype, this vae doesn't work with fp16
	vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
	vae = vae.to(torch.bfloat16).cuda().eval()

	conversation = [
			{
					"role": "User",
					"content": "A stunning princess from kabul in red, white traditional clothing, blue eyes, brown hair",
			},
			{"role": "Assistant", "content": ""},
	]

	sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
			conversations=conversation,
			sft_format=vl_chat_processor.sft_format,
			system_prompt="",
	)
	prompt = sft_format + vl_chat_processor.image_gen_tag

	def set_seed(seed):
			random.seed(seed)
			np.random.seed(seed)
			torch.manual_seed(seed)
			if torch.cuda.is_available():
					torch.cuda.manual_seed(seed)
					torch.cuda.manual_seed_all(seed)
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark = False

	set_seed(42)

	@torch.inference_mode()
	def generate(
			mmgpt: MultiModalityCausalLM,
			vl_chat_processor: VLChatProcessor,
			prompt: str,
			cfg_weight: float = 5.0,
			num_inference_steps: int = 30,
			batchsize: int = 5
	):
			input_ids = vl_chat_processor.tokenizer.encode(prompt)
			input_ids = torch.LongTensor(input_ids)
			
			tokens = torch.stack([input_ids] * 2 * batchsize).cuda()
			tokens[batchsize:, 1:] = vl_chat_processor.pad_id
			inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)

			# we remove the last <bog> token and replace it with t_emb later
			inputs_embeds = inputs_embeds[:, :-1, :] 
			
			# generate with rectified flow ode
			# step 1: encode with vision_gen_enc
			z = torch.randn((batchsize, 4, 48, 48), dtype=torch.bfloat16).cuda()
			
			dt = 1.0 / num_inference_steps
			dt = torch.zeros_like(z).cuda().to(torch.bfloat16) + dt
			
			# step 2: run ode
			attention_mask = torch.ones((2*batchsize, inputs_embeds.shape[1]+577)).to(vl_gpt.device)
			attention_mask[batchsize:, 1:inputs_embeds.shape[1]] = 0
			attention_mask = attention_mask.int()

			for step in range(num_inference_steps):
					# prepare inputs for the llm
					z_input = torch.cat([z, z], dim=0) # for cfg
					t = step / num_inference_steps * 1000.
					t = torch.tensor([t] * z_input.shape[0]).to(dt)
					z_enc = vl_gpt.vision_gen_enc_model(z_input, t)
					z_emb, t_emb, hs = z_enc[0], z_enc[1], z_enc[2]
					z_emb = z_emb.view(z_emb.shape[0], z_emb.shape[1], -1).permute(0, 2, 1)
					z_emb = vl_gpt.vision_gen_enc_aligner(z_emb)
					llm_emb = torch.cat([inputs_embeds, t_emb.unsqueeze(1), z_emb], dim=1)
					# print(llm_emb.shape)

					# input to the llm
					# we apply attention mask for CFG: 1 for tokens that are not masked, 0 for tokens that are masked.
					if step == 0:
							# outputs = vl_gpt.language_model.model(inputs_embeds=llm_emb, 
							# 																 use_cache=True, 
							# 																 attention_mask=attention_mask,
							# 																 past_key_values=None)
							# past_key_values = []
							# for kv_cache in past_key_values:
							# 		k, v = kv_cache[0], kv_cache[1]
							# 		past_key_values.append((k[:, :, :inputs_embeds.shape[1], :], v[:, :, :inputs_embeds.shape[1], :]))
							# past_key_values = tuple(past_key_values)
							past_key_values = None
					else:
							past_key_values = tuple(past_key_values) if past_key_values else None  # Convert only if it's valid
					outputs = vl_gpt.language_model.model(inputs_embeds=llm_emb, 
																						use_cache=True, 
																						attention_mask=attention_mask,
																						past_key_values=past_key_values,
																						mode="gen",
																						timestep=step)
					hidden_states = outputs.last_hidden_state
					
					# transform hidden_states back to v
					hidden_states = vl_gpt.vision_gen_dec_aligner(vl_gpt.vision_gen_dec_aligner_norm(hidden_states[:, -576:, :]))
					hidden_states = hidden_states.reshape(z_emb.shape[0], 24, 24, 768).permute(0, 3, 1, 2)
					v = vl_gpt.vision_gen_dec_model(hidden_states, hs, t_emb)
					v_cond, v_uncond = torch.chunk(v, 2)
					v = cfg_weight * v_cond - (cfg_weight-1.) * v_uncond
					z = z + dt * v
					
			# step 3: decode with vision_gen_dec and sdxl vae
			decoded_image = vae.decode(z / vae.config.scaling_factor).sample
			
			os.makedirs('generated_samples_flow', exist_ok=True)
			save_path = os.path.join('generated_samples_flow', "img.jpg")
			torchvision.utils.save_image(decoded_image.clip_(-1.0, 1.0)*0.5+0.5, save_path)
			print(f"Image saved to {save_path}")

	@torch.inference_mode()
	def generate_kv(
			mmgpt: MultiModalityCausalLM,
			vl_chat_processor: VLChatProcessor,
			prompt: str,
			cfg_weight: float = 5.0,
			num_inference_steps: int = 30,
			batchsize: int = 5
	):
			input_ids = vl_chat_processor.tokenizer.encode(prompt)
			input_ids = torch.LongTensor(input_ids)
			
			tokens = torch.stack([input_ids] * 2 * batchsize).cuda()
			tokens[batchsize:, 1:] = vl_chat_processor.pad_id
			inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)

			# we remove the last <bog> token and replace it with t_emb later
			inputs_embeds = inputs_embeds[:, :-1, :] 
			
			prompt_len = inputs_embeds.shape[1]
			prompt_attention_mask = torch.ones((2*batchsize, prompt_len), dtype=torch.int, device=vl_gpt.device)

			prompt_outputs = vl_gpt.language_model.model(
					inputs_embeds=inputs_embeds,
					use_cache=True,
					attention_mask=prompt_attention_mask,
					past_key_values=None
			)
			prompt_cache = DynamicCache.from_legacy_cache(prompt_outputs.past_key_values)
			
			# generate with rectified flow ode
			# step 1: encode with vision_gen_enc
			torch.manual_seed(42)
			z = torch.randn((batchsize, 4, 48, 48), dtype=torch.bfloat16).cuda()
			
			dt = 1.0 / num_inference_steps
			dt = torch.zeros_like(z).cuda().to(torch.bfloat16) + dt 
			
			# step 2: run ode
			attention_mask = torch.ones((2*batchsize, inputs_embeds.shape[1]+577)).to(vl_gpt.device)
			attention_mask[batchsize:, 1:inputs_embeds.shape[1]] = 0
			attention_mask = attention_mask.int()
			
			for step in range(num_inference_steps):
					# prepare inputs for the llm
					z_input = torch.cat([z, z], dim=0) 
					t = step / num_inference_steps * 1000.
					t = torch.tensor([t] * z_input.shape[0]).to(dt)
					z_enc = vl_gpt.vision_gen_enc_model(z_input, t)
					z_emb, t_emb, hs = z_enc[0], z_enc[1], z_enc[2]
					z_emb = z_emb.view(z_emb.shape[0], z_emb.shape[1], -1).permute(0, 2, 1)
					z_emb = vl_gpt.vision_gen_enc_aligner(z_emb)

					llm_emb_dyn = torch.cat([t_emb.unsqueeze(1), z_emb], dim=1) 

					outputs = vl_gpt.language_model.model(
							inputs_embeds=llm_emb_dyn,
							attention_mask=attention_mask,
							use_cache=True,
							past_key_values=prompt_cache 
					)
					
					past_key_values = []
					for kv_cache in outputs.past_key_values:
							k, v = kv_cache[0], kv_cache[1]
							past_key_values.append((k[:, :, :inputs_embeds.shape[1], :], v[:, :, :inputs_embeds.shape[1], :]))
					prompt_cache = DynamicCache.from_legacy_cache(past_key_values)
					
					hidden_states = outputs.last_hidden_state
					
					# transform hidden_states back to v
					hidden_states = vl_gpt.vision_gen_dec_aligner(vl_gpt.vision_gen_dec_aligner_norm(hidden_states[:, -576:, :]))
					hidden_states = hidden_states.reshape(z_emb.shape[0], 24, 24, 768).permute(0, 3, 1, 2)
					v = vl_gpt.vision_gen_dec_model(hidden_states, hs, t_emb)
					v_cond, v_uncond = torch.chunk(v, 2)
					v = cfg_weight * v_cond - (cfg_weight-1.) * v_uncond
					z = z + dt * v
					
			decoded_image = vae.decode(z / vae.config.scaling_factor).sample
			
			os.makedirs('generated_samples', exist_ok=True)
			save_path = os.path.join('generated_samples', "img2.jpg")
			torchvision.utils.save_image(decoded_image.clip_(-1.0, 1.0)*0.5+0.5, save_path)

	generate(
			vl_gpt,
			vl_chat_processor,
			prompt,
			cfg_weight=2.0,
			num_inference_steps=30,
			batchsize=5
	)

	if base_attention is not None:
		sparsity = base_attention.get_sparsity()
		# --- 格式化输出 ---
		print("\n" + "="*60)
		print(" " * 15 + "Attention Sparsity Summary")
		print("="*60)
		
		# 表头
		print(f"{'Attention Type':<20} | {'Component':<20} | {'Average Sparsity':>15}")
		print("-"*60)

		# 循环处理每种类型
		for cfg_type in ["normal", "cfg_text", "cfg_img"]:
			if cfg_type in sparsity:
				# 计算 VAE + ViT 的稀疏度
				vae_vit_sparsity_list = np.array(sparsity[cfg_type][0])
				if vae_vit_sparsity_list.size > 0: # 确保数组不为空
					avg_vae_vit = np.mean(vae_vit_sparsity_list)
					print(f"{cfg_type.replace('_', ' ').title():<20} | {'VAE + ViT':<20} | {avg_vae_vit:>14.4%}")
				
				# 计算 Self-Attention 的稀疏度
				self_attn_sparsity_list = np.array(sparsity[cfg_type][1])
				if self_attn_sparsity_list.size > 0: # 确保数组不为空
					avg_self_attn = np.mean(self_attn_sparsity_list)
					print(f"{'':<20} | {'Self-Attention':<20} | {avg_self_attn:>14.4%}")
				
				print("-"*60)

	if mlp_args is not None:
		sparsity = mlp_args.mot_sparsity
		# --- 格式化输出 ---
		print("\n" + "="*60)
		print(" " * 15 + "MLP Sparsity Summary")
		print("="*60)
		
		# 表头
		print(f"{'CFG Type':<20} | {'Component':<20} | {'Average Sparsity':>15}")
		print("-"*60)

		# 循环处理每种类型
		for cfg_type in ["cfg_text", "cfg_img"]:
			if cfg_type in sparsity:
				# 计算 MLP 的稀疏度
				mlp_sparsity_list = np.array(sparsity[cfg_type])
				if mlp_sparsity_list.size > 0: # 确保数组不为空
					avg_mlp = np.mean(mlp_sparsity_list)
					print(f"{cfg_type.replace('_', ' ').title():<20} | {'MLP':<20} | {avg_mlp:>14.4%}")
				
				print("-"*60)
