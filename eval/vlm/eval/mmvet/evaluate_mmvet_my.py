# Copyright (c) 2023 OpenGVLab
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under MIT, with the full license text
# available at https://github.com/OpenGVLab/InternVL/blob/main/LICENSE.
#
# This modified file is released under the same license.

# import argparse
# import json
# import os
# import random

# import torch
# from eval.vlm.utils import load_model_and_tokenizer, build_transform, process_conversation
# from PIL import Image
# from tqdm import tqdm

# from modeling.janusflow.janusflow import add_argus_arguments, init_attn_mlp_from_args
# from typing import Dict, Optional

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

import random
from tqdm import tqdm

from eval.vlm.utils import process_conversation

ds_collections = {
		'mmvet': {
				'root': 'eval/vlm/data/mm-vet/images',
				'question': 'eval/vlm/data/mm-vet/llava-mm-vet.jsonl',
				'metric': None,
				'max_new_tokens': 1000,
				'min_new_tokens': 1,
		}
}


class VQADataset(torch.utils.data.Dataset):

		def __init__(self, root, data, prompt):
				self.root = root
				self.data = open(data).readlines()
				self.prompt = prompt
				
		def __len__(self):
				return len(self.data)

		def __getitem__(self, idx):
				data = json.loads(self.data[idx].strip())
				image, question, question_id, annotation = data['image'], data[
						'text'], data['question_id'], data.get('answer', None)

				image = os.path.join(self.root, image)
				image = Image.open(image).convert('RGB')
				images = [image]
				
				question = question + ' ' + self.prompt

				images, conversation = process_conversation(images, question)

				return question_id, question, images, conversation, annotation

def save_metrics_to_jsonl(
		key: str,
		save_path: str,
		text_gen_time: Optional[float] = None,
):
		"""
		将批次指标处理成JSON格式，并追加到JSONL文件中。
		"""
		metrics_record = {"key": key}

		if text_gen_time is not None:
				metrics_record["text_gen_time_ms"] = float(text_gen_time)

		# 如果记录中有任何指标，则写入文件
		if len(metrics_record) > 1:
				with open(save_path, 'a', encoding='utf-8') as f:
						f.write(json.dumps(metrics_record) + '\n')

def evaluate_chat_model(gen_model: JanusFlowModelWrapper):
		# random.seed(args.seed)
		prompt = ''

		for ds_name in args.datasets:
				dataset = VQADataset(
						root=ds_collections[ds_name]['root'],
						data=ds_collections[ds_name]['question'],
						prompt=prompt,
				)

				outputs = {}
				text_gen_times = []
				for _, (question_id, question, images, conversation, annotations) in tqdm(enumerate(dataset)):
						# pred = model.chat(
						# 		tokenizer, 
						# 		new_token_ids,
						# 		image_transform,
						# 		images=images,
						# 		prompt=conversation,
						# 		max_length=ds_collections[ds_name]['max_new_tokens'], # TODO: how to use ds_collections[ds_name]['min_new_tokens']
						# )
						pred, text_gen_time = gen_model.multimodal_understanding(
							image=images, # batch=1
							question=conversation, # batch=1
						)

						outputs[f'v1_{question_id}'] = pred
						text_gen_times.append(text_gen_time)

				print(f'Evaluating {ds_name} ...')
				results_file = os.path.join(args.out_dir, 'results.json')
				json.dump(outputs, open(results_file, 'w'))
				print('Results saved to {}'.format(results_file))

				output_path = os.path.join(args.out_dir, f'text_gen_time.jsonl')
				save_metrics_to_jsonl(
						key="dummy",
						save_path=output_path,
						text_gen_time=np.array(text_gen_times).mean(),
				)

def set_seeds(seed):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
		parser = argparse.ArgumentParser()
		parser.add_argument('--datasets', type=str, default='mmvet')
		parser.add_argument('--batch-size', type=int, default=1)
		parser.add_argument('--num-workers', type=int, default=1)
		parser.add_argument('--out-dir', type=str, default='results')
		parser.add_argument('--seed', type=int, default=0)
		parser.add_argument('--model-path', type=str, default='hf/BAGEL-7B-MoT/')
		parser = add_argus_arguments(parser)
		args = parser.parse_args()

		if not os.path.exists(args.out_dir):
				os.makedirs(args.out_dir, exist_ok=True)

		args.datasets = args.datasets.split(',')
		print('datasets:', args.datasets)
		assert args.batch_size == 1, 'Only batch size 1 is supported'
		
		num_timesteps = 30
		# model, tokenizer, new_token_ids = load_model_and_tokenizer(args)
		# image_transform = build_transform()
		
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

		total_params = sum(p.numel() for p in gen_model.vl_gpt.parameters()) / 1e9
		print(f'[test] total_params: {total_params}B')

		set_seeds(args.seed)
		evaluate_chat_model(gen_model=gen_model)
