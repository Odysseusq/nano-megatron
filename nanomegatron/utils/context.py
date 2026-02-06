from dataclasses import dataclass
import torch


@dataclass
class Context:
    cu_seqlens: torch.Tensor | None = None
    max_seqlen: int = 0

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(cu_seqlens, max_seqlen):
    global _CONTEXT
    _CONTEXT = Context(cu_seqlens, max_seqlen)
