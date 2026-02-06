import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer
from nanomegatron.models.qwen3 import Qwen3ForCausalLM
from nanomegatron.utils.loader import load_model
from nanomegatron.utils.context import set_context
from nanomegatron.optim.grad_accumulator import FP32GradientAccumulator
from nanomegatron.optim.clip_grads import clip_grad_norm
from nanomegatron.optim.lr_scheduler import get_cosine_schedule_with_warmup
from nanomegatron.utils.data import get_dataloader
from nanomegatron.utils.loss import compute_loss
from nanomegatron.config import Config
from nanomegatron.utils.checkpoint import save_checkpoint, load_checkpoint


def get_param_groups(named_params, weight_decay):
    decay, no_decay = [], []
    for name, param in named_params:
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def train(rank, world_size, config: Config):
    dist.init_process_group("nccl", init_method=f"tcp://localhost:{config.parallelism.port}",
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    ckpt = config.checkpoint
    model_path = str(Path(ckpt.resume_from) / "model") if ckpt.resume_from else config.model.path

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hf_config = AutoConfig.from_pretrained(model_path)
    torch.set_default_dtype(hf_config.torch_dtype)
    torch.set_default_device("cuda")

    model = Qwen3ForCausalLM(hf_config)
    load_model(model, model_path)

    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grad_accumulator = FP32GradientAccumulator(named_params)

    opt = config.optimizer
    optimizer_params = grad_accumulator.get_named_parameters_for_optimizer()
    param_groups = get_param_groups(optimizer_params, opt.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=opt.lr, betas=(opt.adam_beta1, opt.adam_beta2), eps=opt.adam_eps)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, opt.warmup_steps, opt.total_steps,
                                                   min_lr_ratio=opt.min_lr / opt.lr)

    start_step = 0
    if ckpt.resume_from:
        start_step = load_checkpoint(model, optimizer, lr_scheduler, grad_accumulator, ckpt.resume_from)

    grad_acc_steps = config.data.gradient_accumulation_steps
    dataloader = get_dataloader(config.data.path, tokenizer, batch_size=config.data.micro_batch_size,
                                seq_len=config.data.seq_len)
    micro_batch_iter = iter(dataloader)
    consumed = start_step * grad_acc_steps

    for step in range(start_step, opt.total_steps):
        loss_acc = 0.0
        for _ in range(grad_acc_steps):
            batch = next(micro_batch_iter)
            consumed += 1
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            positions = batch["positions"].cuda()
            cu_seqlens = batch["cu_seqlens"].cuda()
            max_seqlen = batch["max_seqlen"]

            set_context(cu_seqlens, max_seqlen)

            hidden_states = model(input_ids, positions)
            loss = compute_loss(hidden_states, labels, model.lm_head.weight) / grad_acc_steps
            grad_accumulator.backward(loss)
            loss_acc += loss.item()

        grad_norm = clip_grad_norm(named_params, max_norm=opt.clip_grad, grad_accumulator=grad_accumulator)
        optimizer.step()
        lr_scheduler.step()
        grad_accumulator.step()
        optimizer.zero_grad()
        grad_accumulator.zero_grad()

        if rank == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"step {step}: loss={loss_acc:.4f}, grad_norm={grad_norm.item():.4f}, lr={lr:.2e}")

        if ckpt.save_every and (step + 1) % ckpt.save_every == 0:
            save_checkpoint(model, optimizer, lr_scheduler, grad_accumulator, step + 1,
                            f"{ckpt.save_dir}/step_{step + 1}", config.model.path)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    world_size = config.parallelism.tp
    mp.spawn(train, args=(world_size, config), nprocs=world_size)
