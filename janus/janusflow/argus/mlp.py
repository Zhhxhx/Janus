import torch
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import LlamaMLP
import numpy as np

class MLPArgs:
	def __init__(
		self,
		mlp_backend: str = 'naive',
		nr_timesteps: int = 24,
		nr_layers: int = 16,
	):
		self.mlp_backend = mlp_backend
		self.nr_timesteps = nr_timesteps
		self.nr_layers = nr_layers
		print(f"Using MLP backend: {mlp_backend}")

		self.mot_sparsity = {
			"cfg_text": np.zeros((nr_timesteps ,nr_layers), dtype=np.float32),
			"cfg_img": np.zeros((nr_timesteps, nr_layers), dtype=np.float32)
		}

	def clear_sparsity(self):
		self.mot_sparsity = {
			"cfg_text": np.zeros((self.nr_timesteps ,self.nr_layers), dtype=np.float32),
			"cfg_img": np.zeros((self.nr_timesteps, self.nr_layers), dtype=np.float32)
		}

class WeightGroupQuantizer:
	"""
	一个专门用于对 2D 权重矩阵进行组量化的类。
	支持 INT8, INT4, 和 FP4 模式。
	"""
	def __init__(self, group_size: int = 128, mode: str = 'int8'):
		self.group_size = group_size
		self.mode = mode

		if self.mode == 'int4':
			self.n_bits = 4
			if self.group_size % 2 != 0:
				raise ValueError("对于 INT4 模式, group_size 必须是 2 的倍数。")
		elif self.mode == 'int8':
			self.n_bits = 8
		elif self.mode == 'fp4':
			self.n_bits = 4
			self.fp4_codebook = torch.tensor([
				-1.0, -0.6667, -0.5, -0.3333, -0.25, -0.1667, -0.0833, 0.0,
				0.0833, 0.1667, 0.25, 0.3333, 0.5, 0.6667, 1.0
			], dtype=torch.float16) # 使用 float16 以节省码本存储
		else:
			raise ValueError(f"Unsupported mode: {mode}")

		if 'int' in self.mode:
			self.q_max = 2 ** (self.n_bits - 1) - 1
			self.q_min = -2 ** (self.n_bits - 1)

	def quantize(self, weight: torch.Tensor) -> Tuple:
		if self.mode == 'int8':
			return self._quantize_int(weight)
		elif self.mode == 'int4':
			return self._quantize_int(weight)
		elif self.mode == 'fp4':
			return self._quantize_fp4(weight)
		else:
			raise ValueError(f"Unsupported mode: {self.mode}")

	def dequantize(self, *args, original_shape: Tuple) -> torch.Tensor:
		if self.mode == 'int8':
			return self._dequantize_int(*args, original_shape=original_shape)
		elif self.mode == 'int4':
			q_packed, scales, zero_points = args
			return self._dequantize_int4(q_packed, scales, zero_points, original_shape=original_shape)
		elif self.mode == 'fp4':
			q_indices, scales = args
			return self._dequantize_fp4(q_indices, scales, original_shape=original_shape)
		else:
			raise ValueError(f"Unsupported mode: {self.mode}")

	def _quantize_int(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""INT8 和 INT4 的核心非对称量化逻辑。"""
		assert weight.dim() == 2, "权重必须是 2D 张量"
		out_features, in_features = weight.shape
		if in_features % self.group_size != 0:
			raise ValueError(f"输入特征维度 ({in_features}) 必须能被 group_size ({self.group_size}) 整除。")

		grouped_weight = weight.view(out_features, -1, self.group_size)
		grouped_weight_fp32 = grouped_weight.float()
		
		min_vals, _ = torch.min(grouped_weight_fp32, dim=-1)
		max_vals, _ = torch.max(grouped_weight_fp32, dim=-1)

		scales = (max_vals - min_vals) / (self.q_max - self.q_min)
		scales = scales.clamp(min=1e-6)
		
		zero_points = torch.round(self.q_min - min_vals / scales).to(torch.int8)

		q_weight = torch.round(grouped_weight_fp32 / scales.unsqueeze(-1) + zero_points.unsqueeze(-1))
		q_weight = q_weight.clamp(self.q_min, self.q_max).to(torch.int8)

		if self.mode == 'int4':
			# --- INT4 打包逻辑 ---
			# 将范围 [-8, 7] 映射到 [0, 15]
			q_weight_shifted = q_weight - self.q_min
			# 将每两个 int4 值打包成一个 int8
			q_packed = q_weight_shifted.view(out_features, -1, self.group_size // 2, 2)
			val1 = q_packed[..., 0]
			val2 = q_packed[..., 1]
			# val1 存高4位, val2 存低4位
			q_packed_byte = (val1 << 4) | val2
			return q_packed_byte.to(torch.uint8), scales, zero_points
		else: # INT8
			# print("INT8 quantization completed.")
			return q_weight, scales, zero_points

	def _dequantize_int(self, q_weight: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, original_shape: Tuple) -> torch.Tensor:
		"""INT8 的反量化逻辑。"""
		dequantized_groups = (q_weight.float() - zero_points.unsqueeze(-1)) * scales.unsqueeze(-1)
		return dequantized_groups.reshape(original_shape)

	def _dequantize_int4(self, q_packed: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, original_shape: Tuple) -> torch.Tensor:
		"""INT4 的反量化逻辑，包含解包。"""
		out_features, _ = original_shape
		
		# --- INT4 解包逻辑 ---
		# 从 uint8 解包回两个 int4 值
		val1_shifted = q_packed >> 4
		val2_shifted = q_packed & 0x0F # 掩码，只取低4位
		
		# 组合回原始的 int8 张量形状
		q_weight_shifted = torch.stack([val1_shifted, val2_shifted], dim=-1).view(out_features, -1, self.group_size)
		
		# 从 [0, 15] 映射回 [-8, 7]
		q_weight = q_weight_shifted.to(torch.int8) + self.q_min

		# 使用通用的反量化公式
		return self._dequantize_int(q_weight, scales, zero_points, original_shape)

	def _quantize_fp4(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""FP4 对称量化逻辑。"""
		assert weight.dim() == 2, "权重必须是 2D 张量"
		out_features, in_features = weight.shape
		if in_features % self.group_size != 0:
			raise ValueError(f"输入特征维度 ({in_features}) 必须能被 group_size ({self.group_size}) 整除。")

		grouped_weight = weight.view(out_features, -1, self.group_size)
		
		# 1. 对称量化，只计算 scale
		scales = grouped_weight.abs().max(dim=-1, keepdim=True).values
		scales = scales.clamp(min=1e-6)

		# 2. 归一化到 [-1, 1]
		normalized_weight = grouped_weight / scales

		# 3. 找到码本中最近的值的索引
		codebook = self.fp4_codebook.to(weight.device, dtype=weight.dtype)
		# 扩展维度以进行广播和距离计算
		# normalized_weight: (O, G_num, G_size, 1)
		# codebook:          (1, 1,     1,      C_size)
		abs_diff = torch.abs(normalized_weight.unsqueeze(-1) - codebook)
		q_indices = torch.argmin(abs_diff, dim=-1).to(torch.int8)

		return q_indices, scales.squeeze(-1)

	def _dequantize_fp4(self, q_indices: torch.Tensor, scales: torch.Tensor, original_shape: Tuple) -> torch.Tensor:
		"""FP4 反量化逻辑。"""
		codebook = self.fp4_codebook.to(scales.device, dtype=scales.dtype)
		
		# 1. 使用索引从码本中恢复归一化的值
		quantized_normalized = codebook[q_indices.long()]
		
		# 2. 乘以 scale 恢复浮点值
		dequantized_groups = quantized_normalized * scales.unsqueeze(-1)
		
		return dequantized_groups.reshape(original_shape)

	def simulate_quantization(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		执行量化和立即反量化的往返过程，以模拟精度损失。

		Args:
				weight (torch.Tensor): 原始的浮点权重张量。

		Returns:
				Tuple[torch.Tensor, torch.Tensor]:
				- dequantized_weight: 模拟量化后的浮点权重张量。
				- scales: 计算出的量化尺度，可用于分析。
		"""
		original_shape = weight.shape
		
		# 1. 执行“真实”量化
		quantized_data = self.quantize(weight)
		
		# 2. 立即执行反量化
		dequantized_weight = self.dequantize(*quantized_data, original_shape=original_shape)
		
		# 提取 scales 用于返回
		# scales 通常是元组中的第二个元素
		scales = quantized_data[1]
		
		return dequantized_weight, scales

class TilingUpLinear:
	def __init__(self):
		pass

	def forward(self, act: torch.Tensor, weight: torch.Tensor, nr_head: int) -> torch.Tensor:
		N, D = act.shape
		H = nr_head
		d = D // H
		I, _ = weight.shape
		original_dtype = act.dtype

		act_reshaped = act.view(N, H, d)
		weight_reshaped = weight.T.reshape(H, d, I)

		act_bmm = act_reshaped.permute(1, 0, 2)
		weight_bmm = weight_reshaped
		
		tiled_results = torch.bmm(act_bmm, weight_bmm)
		assert tiled_results.shape == (H, N, I), "TilingUpLinear bmm result has incorrect shape."
		
		return tiled_results.to(original_dtype)

class TilingDownLinear:
	def __init__(self):
		pass

	def forward(self, act: torch.Tensor, weight: torch.Tensor, nr_head: int) -> Tuple[torch.Tensor, torch.Tensor]:
		N, I = act.shape
		D, _ = weight.shape
		H = nr_head
		d = D // H
		original_dtype = act.dtype

		weight_t_chunks = torch.chunk(weight.T, chunks=H, dim=1)

		tiled_results = []
		for chunk in weight_t_chunks:
				tiled_results.append(act @ chunk)
		
		result_concat = torch.cat(tiled_results, dim=1)
		assert result_concat.shape == (N, D), "TilingDownLinear final result has incorrect shape."

		tiled_results_tensor = torch.stack(tiled_results, dim=0)

		return result_concat.to(original_dtype), tiled_results_tensor.to(original_dtype)


class ReuseMLP(LlamaMLP):
	def __init__(
		self, 
		config,
		mlp_backend: str = "argus",
		nr_timesteps: int = 24,
		layer_idx: int = -1,
	):
		super().__init__(config)
		self.tiling_up_proj = TilingUpLinear()
		self.tiling_gate_proj = TilingUpLinear()
		self.tiling_down_proj = TilingDownLinear()

		self.normal_act_cache = None
		self.nr_head = config.num_attention_heads
		assert self.nr_head == 16

		self.cos_threshold = 0.99
		self.quant_type_und = "int8"
		self.quant_type_gen = "int4"
		self.mlp_backend = mlp_backend

		if self.mlp_backend == "argus":
			self.cos_threshold = 0.99
		elif self.mlp_backend == "flightvgm":
			self.cos_threshold = 0.8
		elif self.mlp_backend == "figna":
			self.quant_type_und = "int4"
		elif self.mlp_backend == "axcore":
			self.quant_type_und = "fp4"
			self.quant_type_gen = "fp4"
		elif self.mlp_backend == "default":
			pass
		else:
			raise ValueError(f"Invalid MLP backend: {self.mlp_backend}")

		self.quantizer_und = WeightGroupQuantizer(group_size=32, mode=self.quant_type_und)
		self.quantizer_gen = WeightGroupQuantizer(group_size=32, mode=self.quant_type_gen)

		self.gate_weight_q = None
		self.gate_weight_s = None
		self.up_weight_q = None
		self.up_weight_s = None
		self.down_weight_q = None
		self.down_weight_s = None

		self.layer_idx = layer_idx

	def compute_cosine_similarity(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> torch.Tensor:
		assert tensor_a.shape == tensor_b.shape, "Input tensors must have the same shape."
		N, D = tensor_a.shape
		H = self.nr_head
		d = D // H
		tensor_a = tensor_a.to(torch.float32)
		tensor_b = tensor_b.to(torch.float32)

		tensor_a_reshaped = tensor_a.view(N, H, d).permute(1, 0, 2)  # (H, N, d)
		tensor_b_reshaped = tensor_b.view(N, H, d).permute(1, 0, 2)  # (H, N, d)

		numerator = (tensor_a_reshaped * tensor_b_reshaped).sum(dim=-1)  # (H, N)
		denominator = torch.norm(tensor_a_reshaped, dim=-1) * torch.norm(tensor_b_reshaped, dim=-1)  # (H, N)
		denominator = torch.where(denominator == 0, torch.tensor(1e-8, device=denominator.device), denominator)
		cosine_similarity = numerator / denominator

		return cosine_similarity

	def gen_similarity_mask(self, full_hidden_state: torch.Tensor, cfg_hidden_state: torch.Tensor):
		cosine_sim = self.compute_cosine_similarity(cfg_hidden_state, full_hidden_state)  # (H, B*L)
		similarity_mask = cosine_sim > self.cos_threshold  # (H, B*L)
		return similarity_mask
	
	def reset_und_weight_cache(self):
		self.gate_weight_q, self.gate_weight_s = self.quantizer_und.simulate_quantization(self.gate_proj.weight.data)
		self.up_weight_q, self.up_weight_s = self.quantizer_und.simulate_quantization(self.up_proj.weight.data)
		self.down_weight_q, self.down_weight_s = self.quantizer_und.simulate_quantization(self.down_proj.weight.data)
		
		self.gate_weight_q = self.gate_weight_q.to(device=self.gate_proj.weight.device, dtype=self.gate_proj.weight.dtype)
		self.up_weight_q = self.up_weight_q.to(device=self.up_proj.weight.device, dtype=self.up_proj.weight.dtype)
		self.down_weight_q = self.down_weight_q.to(device=self.down_proj.weight.device, dtype=self.down_proj.weight.dtype)

	def get_gen_quant_weight(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		gate_weight_q, _ = self.quantizer_gen.simulate_quantization(self.gate_proj.weight.data)
		up_weight_q, _ = self.quantizer_gen.simulate_quantization(self.up_proj.weight.data)
		down_weight_q, _ = self.quantizer_gen.simulate_quantization(self.down_proj.weight.data)

		gate_weight_q = torch.cat((gate_weight_q[:gate_weight_q.shape[0]//2, :], self.gate_proj.weight[self.gate_proj.weight.shape[0]//2:, :]), dim=0)
		assert gate_weight_q.shape == self.gate_proj.weight.shape, "Gate projection quantized weight shape mismatch."
		up_weight_q = torch.cat((up_weight_q[:up_weight_q.shape[0]//2, :], self.up_proj.weight[self.up_proj.weight.shape[0]//2:, :]), dim=0)
		assert up_weight_q.shape == self.up_proj.weight.shape, "Up projection quantized weight shape mismatch."
		down_weight_q = torch.cat((down_weight_q[:down_weight_q.shape[0]//2, :], self.down_proj.weight[self.down_proj.weight.shape[0]//2:, :]), dim=0)
		assert down_weight_q.shape == self.down_proj.weight.shape, "Down projection quantized weight shape mismatch."

		gate_weight_q = gate_weight_q.to(device=self.gate_proj.weight.device, dtype=self.gate_proj.weight.dtype)
		up_weight_q = up_weight_q.to(device=self.up_proj.weight.device, dtype=self.up_proj.weight.dtype)
		down_weight_q = down_weight_q.to(device=self.down_proj.weight.device, dtype=self.down_proj.weight.dtype)

		return gate_weight_q, up_weight_q, down_weight_q

	def get_gen_full_quant_weight(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		gate_weight_q, _ = self.quantizer_gen.simulate_quantization(self.gate_proj.weight.data)
		up_weight_q, _ = self.quantizer_gen.simulate_quantization(self.up_proj.weight.data)
		down_weight_q, _ = self.quantizer_gen.simulate_quantization(self.down_proj.weight.data)

		gate_weight_q = gate_weight_q.to(device=self.gate_proj.weight.device, dtype=self.gate_proj.weight.dtype)
		up_weight_q = up_weight_q.to(device=self.up_proj.weight.device, dtype=self.up_proj.weight.dtype)
		down_weight_q = down_weight_q.to(device=self.down_proj.weight.device, dtype=self.down_proj.weight.dtype)

		return gate_weight_q, up_weight_q, down_weight_q

	def argus_forward(
		self, 
		hidden_state: torch.Tensor,
		*,
		mode: Optional[str] = None,
		timestep: Optional[int] = None,
		sparsity: Optional[np.ndarray] = None,
	) -> torch.Tensor:
		B, L, D = hidden_state.shape
		N = B * L
		original_dtype = hidden_state.dtype

		if mode == "und":
			if self.gate_weight_q is None:
				self.reset_und_weight_cache()
			
			assert self.gate_weight_q is not None, "Quantized gate projection weights are not initialized."
			assert self.up_weight_q is not None, "Quantized up projection weights are not initialized."
			assert self.down_weight_q is not None, "Quantized down projection weights are not initialized."
			# print("Using cached quantized weights for 'und' mode.")

			gated_o = torch.nn.functional.linear(hidden_state, self.gate_weight_q)
			up_o = torch.nn.functional.linear(hidden_state, self.up_weight_q)
			gated_value = self.act_fn(gated_o) * up_o
			return torch.nn.functional.linear(gated_value, self.down_weight_q)
		elif mode == "gen":
			full_cond = hidden_state[0:B//2, :, :]
			cfg_cond = hidden_state[B//2:B, :, :]
			full_cond = full_cond.reshape(N//2, D)
			cfg_cond = cfg_cond.reshape(N//2, D)

			gate_weight_q, up_weight_q, down_weight_q = self.get_gen_quant_weight()

			similarity_mask = self.gen_similarity_mask(full_hidden_state=full_cond, cfg_hidden_state=cfg_cond)
			assert similarity_mask.shape == (self.nr_head, N//2), "Similarity mask has incorrect shape."
			assert self.layer_idx != -1, "Layer index is not provided."
			assert timestep is not None, "Timestep is not provided."
			assert sparsity is not None, "Sparsity is not provided."
			sparsity["cfg_img"][timestep, self.layer_idx] = similarity_mask.float().mean().item()

			gated_full_cond_tiling_o = self.tiling_gate_proj.forward(full_cond, gate_weight_q, self.nr_head)
			gated_cfg_cond_tiling_o = self.tiling_gate_proj.forward(cfg_cond, gate_weight_q, self.nr_head)
			gated_cfg_cond_tiling_o = torch.where(
				similarity_mask.unsqueeze(-1),
				gated_full_cond_tiling_o,
				gated_cfg_cond_tiling_o,
			)
			up_full_cond_tiling_o = self.tiling_up_proj.forward(full_cond, up_weight_q, self.nr_head)
			up_cfg_cond_tiling_o = self.tiling_up_proj.forward(cfg_cond, up_weight_q, self.nr_head)
			up_cfg_cond_tiling_o = torch.where(
				similarity_mask.unsqueeze(-1),
				up_full_cond_tiling_o,
				up_cfg_cond_tiling_o,
			)
			gated_full_cond_o = torch.sum(gated_full_cond_tiling_o, dim=0)
			gated_cfg_cond_o = torch.sum(gated_cfg_cond_tiling_o, dim=0)
			gated_full_cond_act_o = self.act_fn(gated_full_cond_o)
			gated_cfg_cond_act_o = self.act_fn(gated_cfg_cond_o)

			up_full_cond_o = torch.sum(up_full_cond_tiling_o, dim=0)
			up_cfg_cond_o = torch.sum(up_cfg_cond_tiling_o, dim=0)

			gated_value = torch.cat([
				gated_full_cond_act_o * up_full_cond_o,
				gated_cfg_cond_act_o * up_cfg_cond_o,
			], dim=0).to(dtype=hidden_state.dtype) #(N, I)
			
			down_o = torch.nn.functional.linear(gated_value, down_weight_q)
			assert down_o.shape == (N, D), "Down projection output has incorrect shape."

			return down_o.view(B, L, -1)
		else:
			raise ValueError(f"Invalid mode: {mode}")

	def _compute_token_pairwise_similarity(self, tokens: torch.Tensor) -> torch.Tensor:
		"""计算一个张量中所有 token 对之间的余弦相似度。"""
		# tokens shape: (B, L, D)
		tokens_fp32 = tokens.to(torch.float32)
		
		# 归一化每个 token 向量
		norm = torch.norm(tokens_fp32, p=2, dim=-1, keepdim=True)
		normalized_tokens = tokens_fp32 / norm.clamp(min=1e-8)
		
		# 通过矩阵乘法计算所有对的余弦相似度
		similarity_matrix = torch.matmul(normalized_tokens, normalized_tokens.transpose(-1, -2)) # (B, L, L)
		
		return similarity_matrix

	def flightvgm_forward(
		self,
		hidden_state: torch.Tensor,
		*,
		mode: Optional[str] = None,
		timestep: Optional[int] = None,
		sparsity: Optional[np.ndarray] = None,
	) -> torch.Tensor:
		B, L, D = hidden_state.shape
		N = B * L

		if mode == "und":
			return super().forward(hidden_state)
		elif mode == "gen":
			device = hidden_state.device
			original_dtype = hidden_state.dtype

			similarity_matrix = self._compute_token_pairwise_similarity(hidden_state) # (B, L, L)

			# 创建一个布尔掩码，标记出下三角和对角线
			mask = torch.ones_like(similarity_matrix, dtype=torch.bool).tril(diagonal=0)
			# 只在掩码为 True 的位置填充 -1
			similarity_matrix.masked_fill_(mask, -1)

			# 找到每个 token (j) 第一个与之相似的 token (i)，其中 i < j
			# `torch.max` 在 dim=1 上操作，为每个批次中的每个 token j 找到最相似的源 token i
			# similar_scores 和 source_indices 的形状都为 (B, L)
			similar_scores, source_indices = torch.max(similarity_matrix, dim=1)

			# 创建一个重用映射，默认为每个 token 计算自己
			# reuse_map[b, j] = i 表示在批次 b 中，token j 将重用 token i 的计算结果
			reuse_map = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

			# 找到那些相似度高于阈值的 token
			reuse_mask = similar_scores > self.cos_threshold

			# 对于需要重用的 token，将它们的映射指向源 token
			# 使用 advanced indexing 需要确保维度匹配
			reuse_map = torch.where(reuse_mask, source_indices, reuse_map)

			unique_indices = torch.unique(reuse_map, dim=1)
			assert sparsity is not None, "Sparsity dictionary is not provided."
			assert self.layer_idx != -1, "Layer index is not provided."
			assert timestep is not None, "Timestep is not provided."
			sparsity["cfg_img"][timestep, self.layer_idx] = np.array([(L - len(i)) / L for i in unique_indices]).mean()

			# 找出所有需要进行独立计算的 token 的索引
			# 由于 reuse_map 是 (B, L)，我们需要一种方法来收集这些 token
			# 一个简单的方法是计算所有 proj，然后使用 gather
			gated_proj_ref = self.gate_proj.forward(hidden_state) # (B, L, I)
			up_proj_ref = self.up_proj.forward(hidden_state)     # (B, L, I)

			# 扩展 reuse_map 以便在最后一个维度上进行 gather
			# reuse_map: (B, L) -> (B, L, 1) -> (B, L, I)
			expanded_reuse_map = reuse_map.unsqueeze(-1).expand(-1, -1, gated_proj_ref.shape[-1])

			# 使用 gather 从参考输出中选取结果
			gated_o = torch.gather(gated_proj_ref, 1, expanded_reuse_map)
			up_o = torch.gather(up_proj_ref, 1, expanded_reuse_map)

			assert gated_o.shape == (B, L, self.gate_proj.out_features), "Gated output shape mismatch."
			assert up_o.shape == (B, L, self.up_proj.out_features), "Up output shape mismatch."

			gated_act_o = self.act_fn(gated_o)
			gated_value = (gated_act_o * up_o).to(dtype=original_dtype)
			down_o = self.down_proj.forward(gated_value)
			return down_o
		else:
			raise ValueError(f"Invalid mode: {mode}")

	def figna_forward(
		self, 
		hidden_state: torch.Tensor,
		*,
		mode: Optional[str] = None,
		timestep: Optional[int] = None,
		sparsity: Optional[np.ndarray] = None,
	):
		if mode == "und":
			if self.gate_weight_q is None:
				self.reset_und_weight_cache()
			gated_o = torch.nn.functional.linear(hidden_state, self.gate_weight_q)
			up_o = torch.nn.functional.linear(hidden_state, self.up_weight_q)
			gated_value = self.act_fn(gated_o) * up_o
			return torch.nn.functional.linear(gated_value, self.down_weight_q)
		elif mode == "gen":
			gated_weight_q, up_weight_q, down_weight_q = self.get_gen_full_quant_weight()

			gated_o = torch.nn.functional.linear(hidden_state, gated_weight_q)
			up_o = torch.nn.functional.linear(hidden_state, up_weight_q)
			gated_value = self.act_fn(gated_o) * up_o
			return torch.nn.functional.linear(gated_value, down_weight_q)
		else:
			raise ValueError(f"Invalid mode: {mode}")

	def forward(
		self, 
		x,
		*,
		mode: Optional[str] = None,
		timestep: Optional[int] = None,
		sparsity: Optional[np.ndarray] = None,
	):
		if self.mlp_backend == "argus":
			return self.argus_forward(x, mode=mode, timestep=timestep, sparsity=sparsity)
		elif self.mlp_backend == "flightvgm":
			return self.flightvgm_forward(x, mode=mode, timestep=timestep, sparsity=sparsity)
		elif self.mlp_backend == "figna":
			return self.figna_forward(x, mode=mode, timestep=timestep, sparsity=sparsity)
		elif self.mlp_backend == "axcore":
			return self.figna_forward(x, mode=mode, timestep=timestep, sparsity=sparsity)
		elif self.mlp_backend == "default":
			return super().forward(x)
		else:
			raise ValueError(f"Invalid MLP backend: {self.mlp_backend}")