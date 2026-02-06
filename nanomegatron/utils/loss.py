import torch
import torch.nn.functional as F
import torch.distributed as dist
from nanomegatron.layers.dist_primitives import differentiable_identity


class ShardedCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, targets, tp_rank, vpp):
        max_logits = logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(max_logits, op=dist.ReduceOp.MAX)
        shifted = logits - max_logits

        exp_shifted = shifted.exp()
        sum_exp = exp_shifted.sum(dim=-1, keepdim=True)
        dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM)
        softmax = exp_shifted / sum_exp

        local_targets = targets - tp_rank * vpp
        mask = (local_targets >= 0) & (local_targets < vpp)
        safe_targets = local_targets.clamp(0, vpp - 1)

        predicted = shifted.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
        predicted = predicted * mask.float()
        dist.all_reduce(predicted, op=dist.ReduceOp.SUM)

        loss = (sum_exp.squeeze(-1).log() - predicted).mean()
        ctx.save_for_backward(softmax, safe_targets, mask)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        softmax, safe_targets, mask = ctx.saved_tensors
        grad = softmax.clone()
        grad.scatter_add_(1, safe_targets.unsqueeze(1), -mask.float().unsqueeze(1))
        return grad * (grad_output / softmax.size(0)), None, None, None


def compute_loss(hidden_states, labels, lm_head_weight):
    tp_size = dist.get_world_size()
    hidden_states = differentiable_identity(hidden_states)
    logits = F.linear(hidden_states, lm_head_weight).float()
    if tp_size > 1:
        return ShardedCrossEntropy.apply(logits, labels, dist.get_rank(), logits.size(-1))
    return F.cross_entropy(logits, labels)
