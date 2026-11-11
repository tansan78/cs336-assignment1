
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class TransformerConfig:
    d_model: int = 1024
    d_ff: int = 512
    num_heads: int = 16

    num_layers: int = 32

    max_seq_len: int = 1024
    rope_theta: float = 1e4
    vocab_size: int = None


@dataclass
class OptimizerConfig:
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    lr: float = 1e-3
    eps: float =1e-8


@dataclass
class PipelineConfig:
    max_l2_norm: float = 1.0
    batch_size: int = 128
    context_length: int = 256

    loss_log_freq: int = 10
    checkpoint_freq: int = 100


@dataclass
class TrainingConfig:
    model_config: TransformerConfig
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig)
    

