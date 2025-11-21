import os
import json
import PIL.Image
import torch
import numpy as np
import os
import json
import argparse
from datetime import datetime
from PIL import Image
from safetensors.torch import load_file
from janus.janusflow.models import MultiModalityCausalLM, VLChatProcessor
import torchvision
from diffusers.models import AutoencoderKL
from modeling.janusflow.janusflow import JanusFlowModelWrapper

from copy import deepcopy

from modeling.janusflow.janusflow import add_argus_arguments, init_attn_mlp_from_args
from typing import Dict, Optional

def move_generation_input_to_device(generation_input, device):
	# Utility to move all tensors in generation_input to device
	for k, v in generation_input.items():
			if isinstance(v, torch.Tensor):
					generation_input[k] = v.to(device)
	return generation_input


def parse_group_args():
	parser = argparse.ArgumentParser(description="Group-based task processor")

	parser.add_argument('--group_num', type=int, required=True,
											help='Total number of groups (must be a positive integer)')

	parser.add_argument('--group_id', type=int, required=True,
											help='ID of the current group (0-based index)')

	args = parser.parse_args()

	return args


@torch.inference_mode()
def generate_image(
	gen_model: JanusFlowModelWrapper,
	prompt, 
	num_timesteps=50, 
	cfg_scale=10.0, 
	num_images=4, 
):  # 添加device参数

	image_list = []
	with torch.no_grad():
		with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
			image_list, image_gen_time = gen_model.generate_image(
				prompt=prompt,
				guidance=cfg_scale,
				num_inference_steps=num_timesteps,
				batchsize=num_images
			)

	return image_list, image_gen_time


def create_image_grid(images, rows, cols):
	"""Creates a grid of images and returns a single PIL Image."""

	assert len(images) == rows * cols

	width, height = images[0].size
	grid_width = width * cols
	grid_height = height * rows

	grid_image = PIL.Image.new('RGB', (grid_width, grid_height))

	for i, image in enumerate(images):
			x = (i % cols) * width
			y = (i // cols) * height
			grid_image.paste(image, (x, y))

	return grid_image

def save_metrics_to_jsonl(
		key: str,
		save_path: str,
		sparsity_data: Optional[Dict[str, np.ndarray]] = None,
		similarity_data: Optional[Dict[str, np.ndarray]] = None,
		avg_self_attn_data: Optional[Dict[str, np.ndarray]] = None,
		image_gen_time: Optional[float] = None,
):
		"""
		将批次指标处理成JSON格式，并追加到JSONL文件中。
		"""
		metrics_record = {"key": key}

		# --- 1. 处理稀疏度数据 ---
		if sparsity_data:
				metrics_record["sparsity"] = {}
				for cfg_type, data_array in sparsity_data.items():
						# if data_array.size == 0 or np.all(data_array == 0): continue
						mean_values = np.mean(data_array, axis=(1, 2))
						metrics_record["sparsity"][cfg_type] = {
								"vae_vit_sparsity": float(mean_values[0]),
								"self_attn_sparsity": float(mean_values[1])
						}

		# --- 2. 处理（MLP）相似度数据 ---
		if similarity_data:
				metrics_record["mlp_similarity"] = {}
				for cfg_type, data_array in similarity_data.items():
						# if data_array.size == 0 or np.all(data_array == 0): continue
						mean_value = np.mean(data_array)
						metrics_record["mlp_similarity"][cfg_type] = float(mean_value)

		# --- 3. 处理平均自注意力分数 ---
		if avg_self_attn_data:
				metrics_record["avg_self_attn_score"] = {}
				for cfg_type, data_array in avg_self_attn_data.items():
						# if data_array.size == 0 or np.all(data_array == 0): continue
						total_mean = np.mean(data_array)
						metrics_record["avg_self_attn_score"][cfg_type] = float(total_mean)

		if image_gen_time is not None:
				metrics_record["image_gen_time_ms"] = float(image_gen_time)

		# 如果记录中有任何指标，则写入文件
		if len(metrics_record) > 1:
				with open(save_path, 'a', encoding='utf-8') as f:
						f.write(json.dumps(metrics_record) + '\n')

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Generate images using Bagel model.")
	parser.add_argument('--group_num', type=int, required=True,
											help='Total number of groups (must be a positive integer)')
	parser.add_argument('--group_id', type=int, required=True,
											help='ID of the current group (0-based index)')
	parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated images.")
	parser.add_argument("--metadata_file", type=str, required=True, help="JSONL file containing lines of metadata for each prompt.")
	parser.add_argument("--num_images", type=int, default=4)
	parser.add_argument("--batch_size", type=int, default=4)
	parser.add_argument("--cfg_scale", type=float, default=4)
	parser.add_argument("--resolution", type=int, default=1024)
	parser.add_argument("--max_latent_size", type=int, default=64)
	parser.add_argument('--model-path', type=str)
	parser.add_argument('--save_grid', action='store_true', default=True)
	parser.add_argument('--max_mem_per_gpu', type=str)
	parser.add_argument('--dtype', type=str)
	parser.add_argument('--n_samples', type=int, default=None)
	
	parser = add_argus_arguments(parser)
	# parser.add_argument("--threshold", type=float, default=0.5, help="New attention probability threshold to set for sparsity in attention mechanism.")
	# parser.add_argument("--attn_backend", type=str, default="naive_sparse_quant", help="Attention backend to use.")
	# parser.add_argument("--sparse_gsize", type=int, default=1, help="Group size for sparse attention.")
	# parser.add_argument("--vae_vit_sparse", action='store_true', help="Whether to apply sparsity to VAE and ViT attention.")
	# parser.add_argument("--self_attn_sparse", action='store_true', help="Whether to apply sparsity to self-attention.")
	# parser.add_argument("--reorder_method", type=str, default="front", help="Method for reordering rope context.")
	# parser.add_argument("--save_dir", type=str, default=None, help="Directory to save attention probabilities.")
	# parser.add_argument("--mlp_save", type=str, default=None, help="Whether to save MLP activations.")
	# parser.add_argument("--mlp_save_dir", type=str, default="maps/mlp_octupusy_flash", help="Directory to save MLP activations.")
	# parser.add_argument("--use_custom_mlp", action='store_true', help="Whether to use custom MLP modules.")
	# parser.add_argument("--mlp_use_similarity", action='store_true', help="Whether to use similarity-based sparsity in MLP.")
	# parser.add_argument("--quantized_mlp_w", action='store_true', help="Whether to use quantized weights for text generation in custom MLP.")
	# parser.add_argument("--use_full_head_similarity", action='store_true', help="Whether to use full head similarity in custom MLP.")
	args = parser.parse_args()
	
	# attention_backend = args.attn_backend
	# base_attention = None
	# if attention_backend != "flash":
	# 	base_attention = TrickAttention(
	# 		attention_backend=attention_backend,
	# 		sparse_gsize=args.sparse_gsize, sparse_topk=0.2, sparse_threshold=args.threshold if args.threshold is not None else 4e-5,
	# 		quant_gsize=32,
	# 		posterior_truncate_threshold=args.threshold if args.threshold is not None else 4e-5,
	# 		save_dir=args.save_dir if args.save_dir else "attn_probs_qkv_dump_tmp",
	# 		is_save=False, is_plot=False, is_truncate=False,
	# 		plot_dir="plot/sparse_attention_scores", heads_to_plot=[0, 1, 2],
	# 		vae_vit=args.vae_vit_sparse, self_attn=args.self_attn_sparse
	# 	)

	# mlp_args = MLPArgs(
	# 	use_custom_mlp=args.use_custom_mlp,
	# 	use_similarity=args.mlp_use_similarity,
	# 	use_quantized_w=args.quantized_mlp_w,
	# 	use_full_head_similarity=args.use_full_head_similarity,
	# 	save_mlp=args.mlp_save,
	# 	save_dir=args.mlp_save_dir
	# )
	
	seed = 42
	if seed is not None:
			import random
			import numpy as np
			random.seed(seed)
			np.random.seed(seed)
			torch.manual_seed(seed)
			if torch.cuda.is_available():
					torch.cuda.manual_seed(seed)
					torch.cuda.manual_seed_all(seed)
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark = False


	cfg_scale = args.cfg_scale
	num_timesteps = 30

	base_attention, mlp_args = init_attn_mlp_from_args(args, nr_timesteps=num_timesteps, nr_layers=24)

	model_path = args.model_path
	vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
	vl_gpt = MultiModalityCausalLM.from_pretrained(
		model_path, trust_remote_code=True, base_attn=base_attention, mlp_args=mlp_args
	)
	vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")

	gen_model = JanusFlowModelWrapper(
		vl_chat_processor=vl_chat_processor,
		vl_gpt=vl_gpt,
		vae=vae,
		device="cuda:0",
		dtype=torch.bfloat16
	)

	with open(args.metadata_file, "r", encoding="utf-8") as fp:
		metadatas = [json.loads(line) for line in fp]
	total_metadatas = len(metadatas)
	
	# Calculate start and end indices like in gen_images_mp.py
	prompts_per_group = (total_metadatas + args.group_num - 1) // args.group_num
	start = args.group_id * prompts_per_group
	if args.n_samples:
		end = min(start + min(prompts_per_group, args.n_samples), total_metadatas)
	else:
		end = min(start + prompts_per_group, total_metadatas)
	print(f"Group {args.group_id}: Processing {end - start} prompts (indices {start} to {end - 1})")

	for idx in range(start, end):
		metadata = metadatas[idx]
		prompt_key = f"{idx:0>5}"
		outpath = os.path.join(args.output_dir, f"{idx:0>5}")
		os.makedirs(outpath, exist_ok=True)
		prompt = metadata['prompt']
		print(f"Group {args.group_id} processing prompt {idx - start + 1}/{end - start}; {idx} in [{start}, {end})")

		sample_path = os.path.join(outpath, "samples")
		os.makedirs(sample_path, exist_ok=True)

		flag = True
		for idx in range(args.num_images):
			if not os.path.exists(os.path.join(sample_path, f"{idx:05}.png")):
				flag = False
				break
		if flag:
			print(f"Group {args.group_id} skipping generation for prompt: {prompt}")
			continue

		with open(os.path.join(outpath, "metadata.jsonl"), "w", encoding="utf-8") as fp:
			json.dump(metadata, fp)

		image_list = []

		metrics_save_path = os.path.join(outpath, "metrics.jsonl")
		if os.path.exists(metrics_save_path):
				open(metrics_save_path, 'w').close()

		for i in range(args.num_images // args.batch_size):
			tmp_image_list, image_gen_time = generate_image(
				gen_model,
				prompt=prompt,
				cfg_scale=cfg_scale, 
				num_timesteps=num_timesteps,
				num_images=args.batch_size,
			)
			image_list.extend(tmp_image_list)

			batch_key = f"{prompt_key}_batch_{i}"
			save_metrics_to_jsonl(
					key=batch_key,
					save_path=metrics_save_path,
					sparsity_data=base_attention.get_sparsity() if base_attention else None,
					similarity_data=mlp_args.mot_sparsity,
					avg_self_attn_data=None,
					image_gen_time=image_gen_time,
			)
			if base_attention:
				base_attention.clear_sparsity()
			mlp_args.clear_sparsity()

		sample_count = 0
		for sample in image_list:
			sample = sample.crop(sample.getbbox())
			sample.save(os.path.join(sample_path, f"{sample_count:05}.png"))
			sample_count += 1

		if args.save_grid:
			grid_image = create_image_grid(image_list, 2, 2)
			grid_image.save(os.path.join(outpath, "grid.png"))