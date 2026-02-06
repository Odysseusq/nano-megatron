import torch
from torch import nn

from flash_attn import flash_attn_varlen_func
from nanomegatron.utils.context import get_context


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        return flash_attn_varlen_func(q, k, v,
                                      cu_seqlens_q=context.cu_seqlens, cu_seqlens_k=context.cu_seqlens,
                                      max_seqlen_q=context.max_seqlen, max_seqlen_k=context.max_seqlen,
                                      softmax_scale=self.scale, causal=True)
