from janus.janusflow.argus.sparse_quant import BlockSparsify, QueryGroupQuantizer
import numpy as np
import torch
from torch import nn
from transformers.processing_utils import Unpack
from typing import Optional
from transformers.utils.generic import TransformersKwargs
import math

class TrickAttention:
	def __init__(
		self,
		attention_backend: str = "naive",
		sparse_gsize: int = 32,
		sparse_threshold: float = 4e-5,
		quant_gsize: int = 32,
		quant_type: str = "int4",
		nr_timesteps: int = 24,
		nr_layers: int = 16,
	):
		self.attention_backend = attention_backend
		self.sparse_gsize = sparse_gsize
		self.sparse_threshold = sparse_threshold
		self.quant_gsize = quant_gsize
		self.quant_type = quant_type

		self.nr_timesteps = nr_timesteps
		self.nr_layers = nr_layers

		print("Initialize TrickAttention with settings:")
		print(f"  attention_backend: {self.attention_backend}")
		print(f"  sparse_gsize: {self.sparse_gsize}")
		print(f"  sparse_threshold: {self.sparse_threshold}")
		print(f"  quant_gsize: {self.quant_gsize}")
		print(f"  quant_type: {self.quant_type}")

		self.block_sparsify = BlockSparsify(block_size=self.sparse_gsize, threshold=self.sparse_threshold)
		self.block_quantizer = QueryGroupQuantizer(group_size=self.quant_gsize, mode=self.quant_type)

		self.sparsity = {
      "normal": np.zeros((2, nr_timesteps, nr_layers), dtype=np.float32),
      "cfg_img": np.zeros((2, nr_timesteps, nr_layers), dtype=np.float32),
		}

		self.sdma_sign_threshold = 0.45

	def get_sparsity(self):
		return self.sparsity
	
	def clear_sparsity(self):
		self.sparsity = {
			"normal": np.zeros((2, self.nr_timesteps, self.nr_layers), dtype=np.float32),
			"cfg_img": np.zeros((2, self.nr_timesteps, self.nr_layers), dtype=np.float32),
		}

	@staticmethod
	def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
		"""
		This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
		num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
		"""
		batch, num_key_value_heads, slen, head_dim = hidden_states.shape
		if n_rep == 1:
				return hidden_states
		hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
		return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

	def naive_attention(
		self,
		module: nn.Module,
		query: torch.Tensor,
		key: torch.Tensor,
		value: torch.Tensor,
		attention_mask: Optional[torch.Tensor],
		scaling: float,
		dropout: float = 0.0,
		**kwargs: Unpack[TransformersKwargs],	
	):
		key_states = self.repeat_kv(key, module.num_key_value_groups)
		value_states = self.repeat_kv(value, module.num_key_value_groups)
		B, H, L, D = query.shape # Not needed if using scaling

		# print(1111)
		# 1. 使用乘法进行缩放，与 eager 版本保持一致
		attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
		print(f"L: {L}")
		if attention_mask is not None and L > 1:
				causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
				# causal_mask = causal_mask.expand_as(attn_weights)
				# attn_weights[~causal_mask] = float('-inf')
				attn_weights = attn_weights + causal_mask

		# 2. 在计算 softmax 时上转型到 float32，这是最关键的修改
		attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
		# attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
		attn_output = torch.matmul(attn_weights, value_states)
		attn_output = attn_output.transpose(1, 2).contiguous()

		return attn_output, attn_weights

	def naive_sparse_quant_attention_cfg(
		self,
		module: nn.Module,
		query: torch.Tensor,
		key: torch.Tensor,
		value: torch.Tensor,
		attention_mask: Optional[torch.Tensor],
		scaling: float,
		dropout: float = 0.0,
		timestep: Optional[int] = None,
		layer_idx: Optional[int] = None,
		**kwargs: Unpack[TransformersKwargs],
	):
		image_token_len = 576
		key_states = self.repeat_kv(key, module.num_key_value_groups)
		value_states = self.repeat_kv(value, module.num_key_value_groups)
		B, H, L, D = query.shape

		attn_weights = torch.matmul(query, key_states.transpose(2, 3) / math.sqrt(D))
		query_quant = self.block_quantizer.simulate_quantization(query)[0].to(dtype=query.dtype)
		# print(f"query_quant data type: {query_quant.dtype}")
		attn_weights_quant = torch.matmul(query_quant, key_states.transpose(2, 3) / math.sqrt(D))
		# block_sparsifier = BlockSparsify(block_size=10, threshold=4e-5, scaling=scaling)
		self_range = torch.tensor((key_states.size(2) - image_token_len, key_states.size(2)), device=key.device, dtype=torch.long)
		self_mask, _ = self.block_sparsify.sparsify_kv_cache_threshold_self_fine(
			q=query,
			k=key_states,
			self_range=self_range,
		)

		assert timestep is not None 
		assert layer_idx is not None
		self.sparsity["normal"][1][timestep, layer_idx] = self_mask.float().mean().item()
		self.sparsity["cfg_img"][1][timestep, layer_idx] = self_mask.float().mean().item()
		# print(f"Self attention sparse mask sparsity: {self_mask.float().mean().item():.4f}")

		attn_weights[:, :, self_range[0]:self_range[1], self_range[0]:self_range[1]] = torch.where(
			self_mask,
			attn_weights_quant[:, :, self_range[0]:self_range[1], self_range[0]:self_range[1]],
			attn_weights[:, :, self_range[0]:self_range[1], self_range[0]:self_range[1]],
		)
		if attention_mask is not None:
			causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
			causal_mask = causal_mask.expand_as(attn_weights)
			attn_weights[~causal_mask] = float('-inf')

		attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
		attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
		attn_output = torch.matmul(attn_weights, value_states)
		attn_output = attn_output.transpose(1, 2).contiguous()

		return attn_output, attn_weights
	
	def _compute_pair_wise_sign_similarity(self, tensor: torch.Tensor) -> torch.Tensor:
		"""
		Computes pairwise sign similarity for a tensor.
		Args:
				tensor (torch.Tensor): Input tensor of shape (B, H, L, D).
		Returns:
				torch.Tensor: Similarity matrix of shape (B, H, L, L).
		"""
		signs = (torch.sign(tensor) < 0).to(torch.float32)
		# Normalize by head dimension to get a value between -1 and 1
		return torch.matmul(signs, signs.transpose(-2, -1)) / tensor.shape[-1]

	def naive_sdma_attention(
		self,
		module: nn.Module,
		query: torch.Tensor,
		key: torch.Tensor,
		value: torch.Tensor,
		attention_mask: Optional[torch.Tensor],
		scaling: float,
		dropout: float = 0.0,
		timestep: Optional[int] = None,
		layer_idx: Optional[int] = None,
		**kwargs: Unpack[TransformersKwargs],
	):
		image_token_len = 576
		key_states = self.repeat_kv(key, module.num_key_value_groups)
		value_states = self.repeat_kv(value, module.num_key_value_groups)
		B, H, Lq, D = query.shape
		_, _, Lk, _ = key_states.shape

		key_sign_similarity = self._compute_pair_wise_sign_similarity(key_states)  # (B, H, Lk, Lk)
		
		# 2. Mask lower triangle and diagonal to find previous similar tokens
		sign_mask = torch.ones_like(key_sign_similarity, dtype=torch.bool).tril(diagonal=0)
		key_sign_similarity.masked_fill_(sign_mask, -1)
		
		# 3. Find the most similar previous token for each token
		similar_score, source_indices = torch.max(key_sign_similarity, dim=-1)  # (B, H, Lk)
		# print(f"Similar score stats - min: {similar_score.min().item():.4f}, max: {similar_score.max().item():.4f}, mean: {similar_score.mean().item():.4f}")
		
		# 4. Create the reuse map
		sign_reuse_map = torch.arange(Lk, device=key.device).view(1, 1, Lk).expand(B, H, -1)
		reuse_mask = similar_score > self.sdma_sign_threshold
		sign_reuse_map = torch.where(reuse_mask, source_indices, sign_reuse_map) # (B, H, Lk)

		# 5. Gather unique keys and values based on the reuse map
		expanded_reuse_map = sign_reuse_map.unsqueeze(-1).expand(-1, -1, -1, D)
		k_unique = torch.gather(key_states, -2, expanded_reuse_map)    # (B, H, Lk, D)
		v_unique = torch.gather(value_states, -2, expanded_reuse_map)  # (B, H, Lk, D)

		# 6. Calculate sparsity for logging
		# Note: This calculation is an approximation averaged over the batch and heads
		unique_counts = torch.tensor([torch.unique(sign_reuse_map[b, h]).numel() for b in range(B) for h in range(H)], device=key.device, dtype=torch.float32)
		reuse_ratio = (Lk - unique_counts) / Lk
		if timestep is not None and layer_idx is not None:
			self.sparsity["normal"][0][timestep, layer_idx] = reuse_ratio.mean().item()

		# 7. Compute attention scores with unique keys
		attn_weights = torch.matmul(query, k_unique.transpose(-2, -1) / math.sqrt(D))
		
		# Use v_unique for the final projection
		# value_to_use = v_unique
		value_to_use = value_states

		if attention_mask is not None:
			causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
			causal_mask = causal_mask.expand_as(attn_weights)
			attn_weights[~causal_mask] = float('-inf')

		attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
		attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
		
		attn_output = torch.matmul(attn_weights, value_to_use)
		attn_output = attn_output.transpose(1, 2).contiguous()

		return attn_output, attn_weights

	def forward(
		self,
		module: nn.Module,
		query: torch.Tensor,
		key: torch.Tensor,
		value: torch.Tensor,
		attention_mask: Optional[torch.Tensor],
		scaling: float,
		dropout: float = 0.0,
		**kwargs: Unpack[TransformersKwargs],
	):
		if self.attention_backend == "naive":
			return self.naive_attention(
				module,
				query,
				key,
				value,
				attention_mask,
				scaling,
				dropout,
				**kwargs,
			)
		elif self.attention_backend == "naive_sparse_quant_cfg":
			return self.naive_sparse_quant_attention_cfg(
				module,
				query,
				key,
				value,
				attention_mask,
				scaling,
				dropout,
				**kwargs,
			)
		elif self.attention_backend == "naive_sdma":
			return self.naive_sdma_attention(
				module,
				query,
				key,
				value,
				attention_mask,
				scaling,
				dropout,
				**kwargs,
			)
		else:
			raise ValueError(f"Unknown attention backend: {self.attention_backend}")