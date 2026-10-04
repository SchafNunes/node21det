"""Configuração de uma execução, lida de YAML.

Caminhos de dados e de saída não ficam no YAML: chegam por argumento de linha
de comando, porque mudam de plataforma para plataforma.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    arch: str = "faster_rcnn"
    pretrained_backbone: bool = True
    trainable_backbone_layers: int = 3
    nms_threshold: float = 0.3
    score_threshold: float = 0.0
    detections_per_image: int = 100
    min_size: int = 800
    max_size: int = 1333
    retinanet_loss_normalization: str = "batch"  # "image" = padrão do torchvision (E9)
    retinanet_head_norm: str = "none"  # "group" = GroupNorm nas torres da cabeça (E10)


@dataclass
class OptimConfig:
    lr: float = 5e-3
    momentum: float = 0.9
    weight_decay: float = 5e-4
    warmup_factor: float = 1e-3  # aquecimento linear na primeira época, como no baseline
    lr_step_epochs: int = 10
    lr_gamma: float = 0.1
    grad_clip_norm: float | None = None  # recorte da norma do gradiente (E10)


@dataclass
class TrainConfig:
    max_epochs: int = 30
    batch_size: int = 6
    num_workers: int = 2
    amp: bool = True
    early_stopping_patience: int = 5
    seed: int = 42


@dataclass
class RunConfig:
    name: str = "run"
    enhancement: str = "none"
    enhancement_params: dict = field(default_factory=dict)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    augmentation: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return RunConfig(
        name=raw.get("name", Path(path).stem),
        enhancement=raw.get("enhancement", "none"),
        enhancement_params=raw.get("enhancement_params", {}),
        model=ModelConfig(**raw.get("model", {})),
        optim=OptimConfig(**raw.get("optim", {})),
        train=TrainConfig(**raw.get("train", {})),
        augmentation=raw.get("augmentation", {}),
    )
