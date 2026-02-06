from dataclasses import dataclass, fields
import yaml


def from_dict(cls, data):
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        v = data[f.name]
        if hasattr(f.type, '__dataclass_fields__') if isinstance(f.type, type) else False:
            v = from_dict(f.type, v)
        elif isinstance(f.type, type) and f.type in (int, float, str, bool) and v is not None:
            v = f.type(v)
        kwargs[f.name] = v
    return cls(**kwargs)


@dataclass
class ModelConfig:
    path: str


@dataclass
class DataConfig:
    path: str
    micro_batch_size: int = 2
    seq_len: int = 2048
    gradient_accumulation_steps: int = 2


@dataclass
class OptimizerConfig:
    lr: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    clip_grad: float = 1.0
    warmup_steps: int = 100
    total_steps: int = 2000


@dataclass
class ParallelismConfig:
    tp: int = 1
    port: int = 2333


@dataclass
class CheckpointConfig:
    save_dir: str = "checkpoints"
    save_every: int = 500
    resume_from: str = None


@dataclass
class Config:
    model: ModelConfig
    data: DataConfig
    optimizer: OptimizerConfig = None
    parallelism: ParallelismConfig = None
    checkpoint: CheckpointConfig = None

    def __post_init__(self):
        if self.optimizer is None:
            self.optimizer = OptimizerConfig()
        if self.parallelism is None:
            self.parallelism = ParallelismConfig()
        if self.checkpoint is None:
            self.checkpoint = CheckpointConfig()

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            d = yaml.safe_load(f)
        return from_dict(cls, d)
