import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file


def _shard_tensor(tensor, tp_dim, tp_sub_sizes, tp_rank, tp_size):
    if tp_dim is None or tp_size == 1:
        return tensor
    if tp_sub_sizes:
        full_sub_sizes = [s * tp_size for s in tp_sub_sizes]
        parts = tensor.split(full_sub_sizes, dim=tp_dim)
        sharded_parts = []
        for part in parts:
            shard_size = part.size(tp_dim) // tp_size
            sharded_parts.append(part.narrow(tp_dim, tp_rank * shard_size, shard_size).contiguous())
        return torch.cat(sharded_parts, dim=tp_dim)
    shard_size = tensor.size(tp_dim) // tp_size
    return tensor.narrow(tp_dim, tp_rank * shard_size, shard_size).contiguous()


def _get_parent_module(model, param_name):
    parts = param_name.rsplit(".", 1)
    return model.get_submodule(parts[0]) if len(parts) > 1 else model


def _build_optimizer_index_to_name(optimizer, grad_accumulator):
    param_id_to_name = {id(p): name for name, p in grad_accumulator.get_named_parameters_for_optimizer()}
    index_to_name = {}
    idx = 0
    for group in optimizer.param_groups:
        for param in group["params"]:
            index_to_name[idx] = param_id_to_name[id(param)]
            idx += 1
    return index_to_name


def _build_reverse_packed_mapping(model):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    reverse_mapping = {}
    for orig_name, (packed_name, shard_id) in packed_modules_mapping.items():
        reverse_mapping.setdefault(packed_name, []).append((orig_name, shard_id))
    for key in reverse_mapping:
        reverse_mapping[key].sort(key=lambda x: ({"q": 0, "k": 1, "v": 2}.get(x[1], x[1])))
    return reverse_mapping


def save_checkpoint(model, optimizer, lr_scheduler, grad_accumulator, step, save_dir, model_config_path):
    tp_rank = dist.get_rank()
    tp_size = dist.get_world_size()
    save_dir = Path(save_dir)
    reverse_packed = _build_reverse_packed_mapping(model)

    hf_state_dict = {}
    for name, param in model.named_parameters():
        module = _get_parent_module(model, name)
        if hasattr(module, "gather_for_save"):
            full_tensor = module.gather_for_save(param.data)
        else:
            full_tensor = param.data.cpu()

        if tp_rank == 0:
            module_short_name = name.rsplit(".", 2)[-2] if name.count(".") >= 2 else None
            sub_sizes = getattr(module, "tp_sub_sizes", None)
            if module_short_name in reverse_packed and sub_sizes:
                full_sub_sizes = [s * tp_size for s in sub_sizes]
                chunks = full_tensor.split(full_sub_sizes, dim=module.tp_dim)
                for (orig_name, _), chunk in zip(reverse_packed[module_short_name], chunks):
                    hf_state_dict[name.replace(module_short_name, orig_name)] = chunk.contiguous()
            else:
                hf_state_dict[name] = full_tensor.contiguous()

    name_to_module = {name: _get_parent_module(model, name) for name, _ in model.named_parameters()}
    index_to_name = _build_optimizer_index_to_name(optimizer, grad_accumulator)
    opt_state_dict = optimizer.state_dict()
    merged_optimizer_state = {}
    for idx, state in opt_state_dict["state"].items():
        param_name = index_to_name[idx]
        module = name_to_module[param_name]
        if tp_rank == 0:
            merged_optimizer_state[param_name] = {}
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                if hasattr(module, "gather_for_save"):
                    gathered = module.gather_for_save(value)
                else:
                    gathered = value.cpu()
                if tp_rank == 0:
                    merged_optimizer_state[param_name][key] = gathered
            else:
                if tp_rank == 0:
                    merged_optimizer_state[param_name][key] = value.cpu() if isinstance(value, torch.Tensor) else value

    if tp_rank == 0:
        model_dir = save_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        save_file(hf_state_dict, model_dir / "model.safetensors")
        for filename in os.listdir(model_config_path):
            if filename.endswith((".json", ".txt", ".model", ".tiktoken")):
                shutil.copy2(Path(model_config_path) / filename, model_dir / filename)
        torch.save(merged_optimizer_state, save_dir / "optimizer.pt")
        torch.save(lr_scheduler.state_dict(), save_dir / "lr_scheduler.pt")
        (save_dir / "latest_step.txt").write_text(str(step))

    dist.barrier()


def load_checkpoint(model, optimizer, lr_scheduler, grad_accumulator, load_dir):
    tp_rank = dist.get_rank()
    tp_size = dist.get_world_size()
    load_dir = Path(load_dir)

    name_to_module = {name: _get_parent_module(model, name) for name, _ in model.named_parameters()}
    index_to_name = _build_optimizer_index_to_name(optimizer, grad_accumulator)
    name_to_param = {}
    idx = 0
    for group in optimizer.param_groups:
        for param in group["params"]:
            name_to_param[index_to_name[idx]] = param
            idx += 1

    merged_optimizer_state = torch.load(load_dir / "optimizer.pt", map_location="cpu", weights_only=False)
    for param_name, state in merged_optimizer_state.items():
        param = name_to_param[param_name]
        module = name_to_module[param_name]
        tp_dim = getattr(module, "tp_dim", None)
        sub_sizes = getattr(module, "tp_sub_sizes", None)
        new_state = {}
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                new_state[key] = _shard_tensor(value, tp_dim, sub_sizes, tp_rank, tp_size).to(device=param.device, dtype=param.dtype)
            else:
                new_state[key] = value.to(device=param.device) if isinstance(value, torch.Tensor) else value
        optimizer.state[param] = new_state

    lr_state_dict = torch.load(load_dir / "lr_scheduler.pt", map_location="cpu", weights_only=False)
    lr_scheduler.load_state_dict(lr_state_dict)
    if hasattr(lr_scheduler, "_last_lr"):
        for group, lr in zip(optimizer.param_groups, lr_scheduler._last_lr):
            group["lr"] = lr

    return int((load_dir / "latest_step.txt").read_text())
