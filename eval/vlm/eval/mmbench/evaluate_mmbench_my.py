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

import os
local_rank = int(os.getenv('LOCAL_RANK', 0))
cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
device_list = [int(x) for x in cuda_devices.split(',')]
os.environ["CUDA_VISIBLE_DEVICES"] = str(device_list[local_rank])

import argparse
import base64
import itertools
import json
import random
from io import BytesIO

import pandas as pd
import torch
from eval.vlm.utils import process_conversation
from PIL import Image
from tqdm import tqdm

from janus.janusflow.models import MultiModalityCausalLM, VLChatProcessor
import torchvision
from diffusers.models import AutoencoderKL
from modeling.janusflow.janusflow import JanusFlowModelWrapper

ds_collections = {
		'mmbench_dev_20230712': {
				'root': 'eval/vlm/data/mmbench/mmbench_dev_20230712.tsv',
				'max_new_tokens': 100,
				'min_new_tokens': 1,
				'type': 'dev',
				'language': 'en'
		},
		'mmbench_dev_cn_20231003': {
				'root': 'eval/vlm/data/mmbench/mmbench_dev_cn_20231003.tsv',
				'max_new_tokens': 100,
				'min_new_tokens': 1,
				'type': 'dev',
				'language': 'cn'
		},
		'mmbench_dev_en_20231003': {
				'root': 'eval/vlm/data/mmbench/mmbench_dev_en_20231003.tsv',
				'max_new_tokens': 100,
				'min_new_tokens': 1,
				'type': 'dev',
				'language': 'en'
		},
		'mmbench_test_cn_20231003': {
				'root': 'eval/vlm/data/mmbench/mmbench_test_cn_20231003.tsv',
				'max_new_tokens': 100,
				'min_new_tokens': 1,
				'type': 'test',
				'language': 'cn'
		},
		'mmbench_test_en_20231003': {
				'root': 'eval/vlm/data/mmbench/mmbench_test_en_20231003.tsv',
				'max_new_tokens': 100,
				'min_new_tokens': 1,
				'type': 'test',
				'language': 'en'
		},
		'ccbench_dev_cn': {
				'root': 'eval/vlm/data/mmbench/CCBench_legacy.tsv',
				'max_new_tokens': 100,
				'min_new_tokens': 1,
				'type': 'dev',
				'language': 'cn'
		}
}


def collate_fn(batches):
		questions = [_['question'] for _ in batches]
		images = [_['images'] for _ in batches]
		conversation = [_['conversation'] for _ in batches]
		answers = [_['answer'] for _ in batches]
		indexes = [_['index'] for _ in batches]
		options = [_['option'] for _ in batches]
		return questions, images, conversation, answers, indexes, options


class MMBenchDataset(torch.utils.data.Dataset):

		def __init__(self, root, prompt, language):
				self.df = pd.read_csv(root, sep='\t')
				self.prompt = prompt
				self.language = language

		def __len__(self):
				return len(self.df)

		def __getitem__(self, idx):
				index = self.df.iloc[idx]['index']
				image = self.df.iloc[idx]['image']
				question = self.df.iloc[idx]['question']
				answer = self.df.iloc[idx]['answer'] if 'answer' in self.df.iloc[0].keys() else None
				# catetory = self.df.iloc[idx]['category']
				# l2_catetory = self.df.iloc[idx]['l2-category']

				image = Image.open(BytesIO(base64.b64decode(image))).convert('RGB')
				images = [image]

				option_candidate = ['A', 'B', 'C', 'D', 'E']
				options = {
						cand: self.load_from_df(idx, cand)
						for cand in option_candidate
						if self.load_from_df(idx, cand) is not None
				}

				hint = self.load_from_df(idx, 'hint')
				if hint is not None:
						question = hint + '\n' + question
				for key, item in options.items():
						question += f'\n{key}. {item}'
				if self.language == 'cn':
						question = question + '\n' + self.prompt['cn']
				else:
						question = question + '\n' + self.prompt['en']

				images, conversation = process_conversation(images, question)

				return {
						'question': question,
						'images': images,
						'conversation': conversation,
						'answer': answer,
						'index': index,
						'option': options
				}

		def load_from_df(self, idx, key):
				if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
						return self.df.iloc[idx][key]
				else:
						return None


class InferenceSampler(torch.utils.data.sampler.Sampler):

		def __init__(self, size):
				self._size = int(size)
				assert size > 0
				self._rank = torch.distributed.get_rank()
				self._world_size = torch.distributed.get_world_size()
				self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

		@staticmethod
		def _get_local_indices(total_size, world_size, rank):
				shard_size = total_size // world_size
				left = total_size % world_size
				shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

				begin = sum(shard_sizes[:rank])
				end = min(sum(shard_sizes[:rank + 1]), total_size)
				return range(begin, end)

		def __iter__(self):
				yield from self._local_indices

		def __len__(self):
				return len(self._local_indices)


def post_process(pred, option):
		pred = pred.strip()
		option_candidate = list(option.keys())
		if len(pred) == 1:
				return pred
		if len(pred) == 0:
				pred = "C"
		elif len(pred) != 1 and pred[0] in option_candidate:
				return pred[0]
		elif len(pred) != 1 and pred[0] not in option_candidate:
				for k, v in option.items():
						if v in pred:
								return k

		return pred


def evaluate_chat_model(gen_model: JanusFlowModelWrapper):
		random.seed(args.seed)

		for ds_name in args.datasets:
				dataset = MMBenchDataset(
						root=ds_collections[ds_name]['root'],
						prompt=prompt,
						language=ds_collections[ds_name]['language'],
				)
				dataloader = torch.utils.data.DataLoader(
						dataset=dataset,
						sampler=InferenceSampler(len(dataset)),
						batch_size=args.batch_size,
						num_workers=args.num_workers,
						pin_memory=True,
						drop_last=False,
						collate_fn=collate_fn,
				)

				outputs = []
				for _, (questions, images, conversation, answers, indexes, options) in tqdm(enumerate(dataloader)):
						# pred = model.chat(
						# 		tokenizer, 
						# 		new_token_ids,
						# 		image_transform,
						# 		images=images[0], # batch=1
						# 		prompt=conversation[0], # batch=1
						# 		max_length=ds_collections[ds_name]['max_new_tokens'], # TODO: how to use ds_collections[ds_name]['min_new_tokens']
						# )
						pred = gen_model.multimodal_understanding(
								image=images[0], # batch=1
								question=conversation[0], # batch=1
						)
						preds = [post_process(pred, options[0])]

						for question, pred, answer, index in zip(questions, preds, answers, indexes):
								outputs.append({
										'question': question,
										'answer': pred,
										'gt_answers': answer,
										'index': int(index)
								})

				torch.distributed.barrier()

				world_size = torch.distributed.get_world_size()
				merged_outputs = [None for _ in range(world_size)]
				torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

				merged_outputs = [json.loads(_) for _ in merged_outputs]
				merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

				if torch.distributed.get_rank() == 0:
						print(f'Evaluating {ds_name} ...')
						results_file = 'results.xlsx'
						output_path = os.path.join(args.out_dir, results_file)
						df = pd.read_table(ds_collections[ds_name]['root'])
						cur_df = df.copy()
						if 'mmbench' in ds_name:
								cur_df = cur_df.drop(columns=['hint', 'category', 'source', 'image', 'comment', 'l2-category'])
								cur_df.insert(6, 'prediction', None)
						else:
								cur_df = cur_df.drop(columns=['category', 'image'])
								cur_df.insert(8, 'prediction', None)
						for item in merged_outputs:
								cur_df.loc[df['index'] == item['index'], 'prediction'] = item['answer']

						cur_df.to_excel(output_path, index=False, engine='openpyxl')
						print('Results saved to {}'.format(output_path))


if __name__ == '__main__':
		parser = argparse.ArgumentParser()
		parser.add_argument('--datasets', type=str, default='mmbench_dev_20230712')
		parser.add_argument('--batch-size', type=int, default=1)
		parser.add_argument('--num-workers', type=int, default=1)
		parser.add_argument('--out-dir', type=str, default='results')
		parser.add_argument('--seed', type=int, default=0)
		parser.add_argument('--model-path', type=str, default='hf/BAGEL-7B-MoT/')
		
		parser.add_argument("--threshold", type=float, default=0.5, help="New attention probability threshold to set for sparsity in attention mechanism.")
		parser.add_argument("--attn_backend", type=str, default="naive_sparse_quant", help="Attention backend to use.")
		parser.add_argument("--sparse_gsize", type=int, default=1, help="Group size for sparse attention.")
		parser.add_argument("--vae_vit_sparse", action='store_true', help="Whether to apply sparsity to VAE and ViT attention.")
		parser.add_argument("--self_attn_sparse", action='store_true', help="Whether to apply sparsity to self-attention.")
		parser.add_argument("--reorder_method", type=str, default="front", help="Method for reordering rope context.")
		parser.add_argument("--save_dir", type=str, default=None, help="Directory to save attention probabilities.")
		parser.add_argument("--mlp_save", type=str, default=None, help="Whether to save MLP activations.")
		parser.add_argument("--mlp_save_dir", type=str, default="maps/mlp_octupusy_flash", help="Directory to save MLP activations.")
		parser.add_argument("--use_custom_mlp", action='store_true', help="Whether to use custom MLP modules.")
		parser.add_argument("--mlp_use_similarity", action='store_true', help="Whether to use similarity-based sparsity in MLP.")
		parser.add_argument("--quantized_mlp_w", action='store_true', help="Whether to use quantized weights for text generation in custom MLP.")
		args = parser.parse_args()

		if not os.path.exists(args.out_dir):
				os.makedirs(args.out_dir, exist_ok=True)

		args.datasets = args.datasets.split(',')
		print('datasets:', args.datasets)
		assert args.batch_size == 1, 'Only batch size 1 is supported'

		# torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
		torch.cuda.set_device(0)

		torch.distributed.init_process_group(
				backend='nccl',
				world_size=int(os.getenv('WORLD_SIZE', '1')),
				rank=int(os.getenv('RANK', '0')),
		)
		
		# attention_backend = args.attn_backend
		# base_attention = None
		# if attention_backend != "flash":
		#   base_attention = TrickAttention(
		#     attention_backend=attention_backend,
		#     sparse_gsize=args.sparse_gsize, sparse_topk=0.2, sparse_threshold=args.threshold if args.threshold is not None else 4e-5,
		#     quant_gsize=32,
		#     posterior_truncate_threshold=args.threshold if args.threshold is not None else 4e-5,
		#     save_dir=args.save_dir if args.save_dir else "attn_probs_qkv_dump_tmp",
		#     is_save=False, is_plot=False, is_truncate=False,
		#     plot_dir="plot/sparse_attention_scores", heads_to_plot=[0, 1, 2],
		#     vae_vit=args.vae_vit_sparse, self_attn=args.self_attn_sparse
		#   )

		# mlp_args = MLPArgs(
		#   use_custom_mlp=args.use_custom_mlp,
		#   use_quantized_w=args.quantized_mlp_w,
		#   use_similarity=args.mlp_use_similarity,
		#   save_mlp=args.mlp_save,
		#   save_dir=args.mlp_save_dir
		# )

		print(f"device = {int(os.getenv('LOCAL_RANK', 0))}")

		# model, _, tokenizer, new_token_ids, _, image_transform = setup_models(
		#     model_path=args.model_path, base_attn=base_attention, mlp_args=mlp_args, device=0
		# )
		# model, tokenizer, new_token_ids = load_model_and_tokenizer(args)
		# model, tokenizer, new_token_ids = load_model_and_tokenizer_my(args, base_attn=base_attention, mlp_args=mlp_args)
		# image_transform = build_transform()

		model_path = args.model_path
		vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
		vl_gpt = MultiModalityCausalLM.from_pretrained(
			model_path, trust_remote_code=True
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

		prompt = {
				'en': "Answer with the option's letter from the given choices directly.",
				'cn': '请直接回答选项字母。'
		}
		evaluate_chat_model(gen_model=gen_model)
