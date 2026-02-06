from collections import OrderedDict
import torch
import torch.distributed as dist
from torch import nn


class FP32GradientAccumulator:
    def __init__(self, named_parameters: list[tuple[str, nn.Parameter]]):
        named_parameters = [(n, p) for n, p in named_parameters if p.requires_grad]

        self.fp32_grad_buffers: dict[str, dict] = OrderedDict()
        self.parameters: dict[str, dict] = {}

        grad_offset = 0
        param_offset = 0
        grad_numel = sum(p.numel() for _, p in named_parameters)
        param_numel = grad_numel

        self._contiguous_grad_buffer = torch.zeros(grad_numel, dtype=torch.float, device="cuda")
        self._contiguous_param_buffer = torch.empty(param_numel, dtype=torch.float, device="cuda")

        for name, param in named_parameters:
            n = param.numel()

            fp32_grad = self._contiguous_grad_buffer[grad_offset:grad_offset + n].view_as(param)
            self.fp32_grad_buffers[name] = {"half": param, "fp32_grad": fp32_grad}
            grad_offset += n

            fp32_param = self._contiguous_param_buffer[param_offset:param_offset + n].view_as(param)
            with torch.no_grad():
                fp32_param.copy_(param)
            fp32_param.requires_grad = True
            self.parameters[name] = {"fp32": fp32_param, "half": param}
            param_offset += n

    def backward(self, loss: torch.Tensor):
        loss.backward()
        for name, elt in self.fp32_grad_buffers.items():
            half_param = elt["half"]
            if half_param.grad is None:
                continue
            fp32_grad = elt["fp32_grad"]
            fp32_grad.add_(half_param.grad)
            half_param.grad = None
            if name in self.parameters:
                self.parameters[name]["fp32"].grad = fp32_grad

    @torch.inference_mode()
    def step(self):
        for name in self.parameters:
            self.parameters[name]["half"].copy_(self.parameters[name]["fp32"])

    def zero_grad(self):
        self._contiguous_grad_buffer.zero_()

    def clip_grad(self, max_norm: float) -> torch.Tensor:
        local_norm_square = self._contiguous_grad_buffer.dot(self._contiguous_grad_buffer)
        dist.all_reduce(local_norm_square, op=dist.ReduceOp.SUM)
        total_norm = local_norm_square.sqrt()
        clip_coef = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
        self._contiguous_grad_buffer.mul_(clip_coef)
        return total_norm

    def get_parameter_for_optimizer(self, name: str) -> nn.Parameter:
        return self.parameters[name]["fp32"]

    def get_grad_buffer(self, name: str) -> torch.Tensor:
        return self.fp32_grad_buffers[name]["fp32_grad"]

    def get_named_parameters_for_optimizer(self):
        return [(name, elt["fp32"]) for name, elt in self.parameters.items()]
