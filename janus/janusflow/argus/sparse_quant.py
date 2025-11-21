import torch
from typing import Tuple
import torch.nn.functional as F
import math

class BlockSparsify:
  def __init__(self, block_size: int = 16, threshold: float = 0.1):
    self.block_size = block_size
    self.threshold = threshold
    self.threshold_max = False

  def padding(self, x: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, int]:
    B, H, L, D = x.shape
    
    padding_len = (group_size - (L % group_size)) % group_size
    
    if padding_len > 0:
      if self.threshold_max:
        pad_value = float('-inf')
        x = F.pad(x, (0, 0, 0, padding_len), value=pad_value)
      else:
        remain_len = L % group_size
        if remain_len == 0:
            remain_len = group_size
            
        last_group = x[:, :, -remain_len:, :]
        pad_value = last_group.mean(dim=2, keepdim=True)
        pad_tensor = pad_value.repeat(1, 1, padding_len, 1)
        x = torch.cat([x, pad_tensor], dim=2)
        
    return x, padding_len
  def sparsify_kv_cache_threshold_self_fine(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    self_range: torch.Tensor,
  ):
    B, H, L, D = q.shape
    _, _, N, _ = k.shape
    assert q.shape[0] == k.shape[0] and q.shape[1] == k.shape[1] and q.shape[3] == k.shape[3], "Q and K must have same B, H, and D dimensions."
    
    group_size = self.block_size

    # 1. 根据 self_range 截取 Q 和 K
    # 截取 self_q 用于计算注意力，L_self 是 self_q 的序列长度
    self_q = q[:, :, self_range[0]:self_range[1], :]
    B, H, L_self, D = self_q.shape
    assert L_self == (self_range[1] - self_range[0]), f"L_self {L_self} must equal to self_range length {(self_range[1] - self_range[0])}"

    pre_self_k = k[:, :, 0:self_range[0], :]
    self_k = k[:, :, self_range[0]:self_range[1], :]
    assert self_range[1] == N, f"self_range end {self_range[1]} must equal to N {N}"

    # 2. 对 Q 和 K 的相关部分进行填充 (Padding)
    self_q_padded, _ = self.padding(self_q, group_size)
    pre_self_k_padded, _ = self.padding(pre_self_k, group_size)
    self_k_padded, self_k_padded_len = self.padding(self_k, group_size)

    # 3. 组合 K，并确定 self_k 在填充后的 K 中的范围
    k_padded = torch.cat([pre_self_k_padded, self_k_padded], dim=2)
    
    self_padded_range = torch.tensor((
        pre_self_k_padded.size(2),
        pre_self_k_padded.size(2) + self_k_padded.size(2) - self_k_padded_len
    ), device=k.device, dtype=torch.long)

    # 4. 计算代表性向量 (Representative Vectors)
    num_groups_q = self_q_padded.size(2) // group_size
    num_groups_k = k_padded.size(2) // group_size
    
    q_padded_grouped = self_q_padded.view(B, H, num_groups_q, group_size, D)
    k_padded_grouped = k_padded.view(B, H, num_groups_k, group_size, D)

    if self.threshold_max:
        representative_q = q_padded_grouped.max(dim=3).values
        representative_k = k_padded_grouped.max(dim=3).values
    else:
        representative_q = q_padded_grouped.mean(dim=3)
        representative_k = k_padded_grouped.mean(dim=3)

    # (B, H, num_groups_q, D) @ (B, H, D, num_groups_k) -> (B, H, num_groups_q, num_groups_k)
    representative_attn_scores = torch.matmul(representative_q, representative_k.transpose(-2, -1) / math.sqrt(D))
    representative_attn_scores = torch.softmax(representative_attn_scores, dim=-1)

    block_mask = representative_attn_scores < self.threshold
    
    # 沿 L 维度放大
    full_mask = torch.repeat_interleave(block_mask, repeats=group_size, dim=2)
    full_mask = torch.repeat_interleave(full_mask, repeats=group_size, dim=3)

    # 6. 截取最终需要的 self_mask
    # 从放大后的完整掩码中，提取出 self_q 和 self_k 对应的部分
    self_mask = full_mask[:, :, 0:L_self, self_padded_range[0]:self_padded_range[1]]
    
    assert self_mask.shape == (B, H, L_self, (self_range[1] - self_range[0])), \
        f"Final mask shape is wrong: {self_mask.shape}"

    return self_mask, representative_attn_scores

class QueryGroupQuantizer:
    """
    一个专门用于对 4D 激活张量 (如 query) 进行组量化的类。
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

    def quantize(self, query: torch.Tensor) -> Tuple:
        if self.mode == 'int8':
            return self._quantize_int(query)
        elif self.mode == 'int4':
            return self._quantize_int(query)
        elif self.mode == 'fp4':
            return self._quantize_fp4(query)
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

    def _quantize_int(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """INT8 和 INT4 的核心非对称量化逻辑。"""
        assert query.dim() == 4, "Query 必须是 4D 张量 (B, H, L, D)"
        B, H, L, D = query.shape
        if D % self.group_size != 0:
            raise ValueError(f"特征维度 ({D}) 必须能被 group_size ({self.group_size}) 整除。")

        grouped_query = query.view(B, H, L, -1, self.group_size)
        grouped_query_fp32 = grouped_query.float()
        
        min_vals, _ = torch.min(grouped_query_fp32, dim=-1)
        max_vals, _ = torch.max(grouped_query_fp32, dim=-1)

        scales = (max_vals - min_vals) / (self.q_max - self.q_min)
        scales = scales.clamp(min=1e-6)
        
        zero_points = torch.round(self.q_min - min_vals / scales).to(torch.int8)

        q_query = torch.round(grouped_query_fp32 / scales.unsqueeze(-1) + zero_points.unsqueeze(-1))
        q_query = q_query.clamp(self.q_min, self.q_max).to(torch.int8)

        if self.mode == 'int4':
            # --- INT4 打包逻辑 ---
            # 将范围 [-8, 7] 映射到 [0, 15]
            q_query_shifted = q_query - self.q_min
            # 将每两个 int4 值打包成一个 int8
            q_packed = q_query_shifted.view(B, H, L, -1, self.group_size // 2, 2)
            val1 = q_packed[..., 0]
            val2 = q_packed[..., 1]
            # val1 存高4位, val2 存低4位
            q_packed_byte = (val1 << 4) | val2
            return q_packed_byte.to(torch.uint8), scales, zero_points
        else: # INT8
            return q_query, scales, zero_points

    def _dequantize_int(self, q_query: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, original_shape: Tuple) -> torch.Tensor:
        """INT8 的反量化逻辑。"""
        dequantized_groups = (q_query.float() - zero_points.unsqueeze(-1)) * scales.unsqueeze(-1)
        return dequantized_groups.reshape(original_shape)

    def _dequantize_int4(self, q_packed: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, original_shape: Tuple) -> torch.Tensor:
        """INT4 的反量化逻辑，包含解包。"""
        B, H, L, D = original_shape
        
        # --- INT4 解包逻辑 ---
        # 从 uint8 解包回两个 int4 值
        val1_shifted = q_packed >> 4
        val2_shifted = q_packed & 0x0F # 掩码，只取低4位
        
        # 组合回原始的 int8 张量形状
        q_query_shifted = torch.stack([val1_shifted, val2_shifted], dim=-1).view(B, H, L, -1, self.group_size)
        
        # 从 [0, 15] 映射回 [-8, 7]
        q_query = q_query_shifted.to(torch.int8) + self.q_min

        # 使用通用的反量化公式
        return self._dequantize_int(q_query, scales, zero_points, original_shape)

    def _quantize_fp4(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """FP4 对称量化逻辑。"""
        assert query.dim() == 4, "Query 必须是 4D 张量 (B, H, L, D)"
        B, H, L, D = query.shape
        if D % self.group_size != 0:
            raise ValueError(f"特征维度 ({D}) 必须能被 group_size ({self.group_size}) 整除。")

        grouped_query = query.view(B, H, L, -1, self.group_size)
        
        # 1. 对称量化，只计算 scale
        scales = grouped_query.abs().max(dim=-1, keepdim=True).values
        scales = scales.clamp(min=1e-6)

        # 2. 归一化到 [-1, 1]
        normalized_query = grouped_query / scales

        # 3. 找到码本中最近的值的索引
        codebook = self.fp4_codebook.to(query.device, dtype=query.dtype)
        # 扩展维度以进行广播和距离计算
        abs_diff = torch.abs(normalized_query.unsqueeze(-1) - codebook)
        q_indices = torch.argmin(abs_diff, dim=-1).to(torch.int8)

        return q_indices, scales.squeeze(-1)

    def _dequantize_fp4(self, q_indices: torch.Tensor, scales: torch.Tensor, original_shape: Tuple) -> torch.Tensor:
        """FP4 反量化逻辑。"""
        codebook = self.fp4_codebook.to(scales.device, dtype=scales.dtype)
        
        # 1. 使用索引从码本中恢复归一化的值
        quantized_normalized = codebook[q_indices]
        
        # 2. 乘以 scale 恢复浮点值
        dequantized_groups = quantized_normalized * scales.unsqueeze(-1)
        
        return dequantized_groups.reshape(original_shape)
    
    def simulate_quantization(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行量化和立即反量化的往返过程，以模拟精度损失。

        Args:
                query (torch.Tensor): 原始的浮点激活张量。

        Returns:
                Tuple[torch.Tensor, torch.Tensor]:
                - dequantized_query: 模拟量化后的浮点激活张量。
                - scales: 计算出的量化尺度，可用于分析。
        """
        original_shape = query.shape
        
        # 1. 执行“真实”量化
        quantized_data = self.quantize(query)
        
        # 2. 立即执行反量化
        dequantized_query = self.dequantize(*quantized_data, original_shape=original_shape)
        
        # 提取 scales 用于返回
        # scales 通常是元组中的第二个元素
        scales = quantized_data[1]
        
        return dequantized_query, scales