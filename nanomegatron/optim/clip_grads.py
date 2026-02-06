import torch


def clip_grad_norm(
    named_parameters,
    max_norm: float,
    grad_accumulator=None,
    norm_type: float = 2.0,
) -> torch.Tensor:
    named_parameters = list(named_parameters)

    if grad_accumulator is not None:
        grads = [grad_accumulator.get_grad_buffer(name) for name, _ in named_parameters]
    else:
        grads = [p.grad for _, p in named_parameters if p.grad is not None]

    if len(grads) == 0:
        return torch.tensor(0.0, device="cuda")

    total_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(g.detach(), ord=norm_type, dtype=torch.float) for g in grads]),
        ord=norm_type,
        dtype=torch.float,
    ).pow(norm_type)

    total_norm.pow_(1.0 / norm_type)

    clip_coef = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    for g in grads:
        g.detach().mul_(clip_coef)

    return total_norm
