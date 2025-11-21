import torch
from janus.janusflow.models import MultiModalityCausalLM, VLChatProcessor
from PIL import Image
import numpy as np
from typing import Optional

import argparse

from janus.janusflow.argus.attention import TrickAttention
from janus.janusflow.argus.mlp import MLPArgs

def add_argus_arguments(parser: argparse.ArgumentParser):
	parser.add_argument("--threshold", type=float, default=4e-5, help="Sparsity threshold for Argus attention.")
	parser.add_argument("--attn_backend", type=str, default="argus", help="Attention backend to use.")
	parser.add_argument("--sparse_gsize", type=int, default=32, help="Sparsity group size for Argus attention.")
	parser.add_argument("--mlp_backend", type=str, default="argus", help="MLP backend to use.")
	return parser

def init_attn_mlp_from_args(args: argparse.Namespace, nr_timesteps: int, nr_layers: int):
	base_attention = None
	if args.attn_backend != "flash":
		base_attention = TrickAttention(
			attention_backend=args.attn_backend,
			sparse_gsize=args.sparse_gsize, 
			sparse_threshold=args.threshold,
			quant_gsize=32,
			quant_type="int4",
			nr_timesteps=nr_timesteps,
			nr_layers=nr_layers,
		)

	mlp_args = MLPArgs(
		mlp_backend=args.mlp_backend,
		nr_timesteps=nr_timesteps,
		nr_layers=nr_layers,
	)

	return base_attention, mlp_args

class JanusFlowModelWrapper:
	def __init__(self, device, dtype, vl_chat_processor: VLChatProcessor, vl_gpt: MultiModalityCausalLM, vae):
		self.device = device
		self.dtype = dtype
		self.vl_chat_processor: VLChatProcessor = vl_chat_processor
		self.tokenizer = vl_chat_processor.tokenizer
		self.vl_gpt: MultiModalityCausalLM = vl_gpt
		self.vl_gpt = self.vl_gpt.to(dtype).to(device).eval()
		self.vae = vae
		self.vae = self.vae.to(dtype).to(device).eval()

	@torch.inference_mode()
	def multimodal_understanding(self, image, question, top_p: Optional[float] = None, temperature: Optional[float] = None):
		# Clear CUDA cache before generating
		torch.cuda.empty_cache()
		
		# set seed
		# torch.manual_seed(seed)
		# np.random.seed(seed)
		# torch.cuda.manual_seed(seed)
		
		conversation = [
				{
						"role": "User",
						"content": f"<image_placeholder>\n{question}",
						"images": [image],
				},
				{"role": "Assistant", "content": ""},
		]
		
		if isinstance(image, Image.Image):
				pil_images = [image]
		elif isinstance(image, list):
				pil_images = image  # 直接使用传入的列表
		elif isinstance(image, torch.Tensor):
				pil_images = [Image.fromarray(image)]
		else:
				raise ValueError("Unsupported image type.")
		prepare_inputs = self.vl_chat_processor(
				conversations=conversation, images=pil_images, force_batchify=True
		).to(self.device, dtype=self.dtype if torch.cuda.is_available() else torch.float16)
		
		
		inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
		
		start_event = torch.cuda.Event(enable_timing=True)
		end_event = torch.cuda.Event(enable_timing=True)
		start_event.record()
		outputs = self.vl_gpt.language_model.generate(
				inputs_embeds=inputs_embeds,
				attention_mask=prepare_inputs.attention_mask,
				pad_token_id=self.tokenizer.eos_token_id,
				bos_token_id=self.tokenizer.bos_token_id,
				eos_token_id=self.tokenizer.eos_token_id,
				max_new_tokens=512,
				do_sample=False if temperature == 0 else True,
				use_cache=True,
				temperature=temperature,
				top_p=top_p,
				# mode="gen",
		)
		end_event.record()
		torch.cuda.synchronize()
		text_gen_time = start_event.elapsed_time(end_event)

		answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

		return answer, text_gen_time
	
	@torch.inference_mode()
	def generate(
		self,
		input_ids,
		cfg_weight: float = 2.0,
		num_inference_steps: int = 30,
		batchsize: int = 5,
	):
		# we generate 5 images at a time, *2 for CFG
		tokens = torch.stack([input_ids] * 2 * batchsize).cuda()
		tokens[batchsize:, 1:] = self.vl_chat_processor.pad_id
		inputs_embeds = self.vl_gpt.language_model.get_input_embeddings()(tokens)
		print(inputs_embeds.shape)

		# we remove the last <bog> token and replace it with t_emb later
		inputs_embeds = inputs_embeds[:, :-1, :] 
		
		# generate with rectified flow ode
		# step 1: encode with vision_gen_enc
		z = torch.randn((batchsize, 4, 48, 48), dtype=torch.bfloat16).cuda()
		
		dt = 1.0 / num_inference_steps
		dt = torch.zeros_like(z).cuda().to(torch.bfloat16) + dt
		
		# step 2: run ode
		attention_mask = torch.ones((2*batchsize, inputs_embeds.shape[1]+577)).to(self.vl_gpt.device)
		attention_mask[batchsize:, 1:inputs_embeds.shape[1]] = 0
		attention_mask = attention_mask.int()

		start_event = torch.cuda.Event(enable_timing=True)
		end_event = torch.cuda.Event(enable_timing=True)
		start_event.record()
		for step in range(num_inference_steps):
			# prepare inputs for the llm
			z_input = torch.cat([z, z], dim=0) # for cfg
			t = step / num_inference_steps * 1000.
			t = torch.tensor([t] * z_input.shape[0]).to(dt)
			z_enc = self.vl_gpt.vision_gen_enc_model(z_input, t)
			z_emb, t_emb, hs = z_enc[0], z_enc[1], z_enc[2]
			z_emb = z_emb.view(z_emb.shape[0], z_emb.shape[1], -1).permute(0, 2, 1)
			z_emb = self.vl_gpt.vision_gen_enc_aligner(z_emb)
			llm_emb = torch.cat([inputs_embeds, t_emb.unsqueeze(1), z_emb], dim=1)

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
			outputs = self.vl_gpt.language_model.model(inputs_embeds=llm_emb, 
																				use_cache=True, 
																				attention_mask=attention_mask,
																				past_key_values=past_key_values,
																				mode="gen",
																				timestep=step)
			hidden_states = outputs.last_hidden_state
			
			# transform hidden_states back to v
			hidden_states = self.vl_gpt.vision_gen_dec_aligner(self.vl_gpt.vision_gen_dec_aligner_norm(hidden_states[:, -576:, :]))
			hidden_states = hidden_states.reshape(z_emb.shape[0], 24, 24, 768).permute(0, 3, 1, 2)
			v = self.vl_gpt.vision_gen_dec_model(hidden_states, hs, t_emb)
			v_cond, v_uncond = torch.chunk(v, 2)
			v = cfg_weight * v_cond - (cfg_weight-1.) * v_uncond
			z = z + dt * v

		end_event.record()
		torch.cuda.synchronize()
		image_gen_time = start_event.elapsed_time(end_event)

		# step 3: decode with vision_gen_dec and sdxl vae
		decoded_image = self.vae.decode(z / self.vae.config.scaling_factor).sample
		
		images = decoded_image.float().clip_(-1., 1.).permute(0,2,3,1).cpu().numpy()
		images = ((images+1) / 2. * 255).astype(np.uint8)
		
		return images, image_gen_time


	@torch.inference_mode()
	def generate_image(
		self,
		prompt,
		seed=None,
		guidance=5.0,
		num_inference_steps=30,
		batchsize=5,
	):
		# Clear CUDA cache and avoid tracking gradients
		torch.cuda.empty_cache()
		# Set the seed for reproducible results
		if seed is not None:
			torch.manual_seed(seed)
			torch.cuda.manual_seed(seed)
			np.random.seed(seed)
		
		with torch.no_grad():
			messages = [{'role': 'User', 'content': prompt},
									{'role': 'Assistant', 'content': ''}]
			text = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(conversations=messages,
																																sft_format=self.vl_chat_processor.sft_format,
																																system_prompt='')
			text = text + self.vl_chat_processor.image_start_tag
			input_ids = torch.LongTensor(self.tokenizer.encode(text))
			images, image_gen_time = self.generate(
				input_ids,
				cfg_weight=guidance,
				num_inference_steps=num_inference_steps,
				batchsize=batchsize,
			)
			return [Image.fromarray(images[i]).resize((1024, 1024), Image.LANCZOS) for i in range(images.shape[0])], image_gen_time
