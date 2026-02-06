import torch
import torch.distributed as dist


class DifferentiableIdentity(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return DifferentiableAllReduceSum.apply(grad_output)


class DifferentiableAllReduceSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        if dist.get_world_size() == 1:
            return tensor
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def differentiable_identity(tensor):
    return DifferentiableIdentity.apply(tensor)

def differentiable_all_reduce_sum(tensor):
    return DifferentiableAllReduceSum.apply(tensor)
