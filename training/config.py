"""Single-run configuration with paths resolved from the repository root."""

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

from hafvillai_runtime.config import ModelSpec

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TrainConfig(ModelSpec):
    role: str
    dataset: str
    output_dir: str
    source_dataset: str = "dataset-studio/data/coser/sft-v1/sft.jsonl"
    prepared_dir: str = "training/data/prepared"
    seed: int = 42
    validation_fraction: float = 0.1
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    epochs: int = 3
    gradient_accumulation: int = 8
    learning_rate: float = 0.0001
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    loss_chunk_tokens: int = 128
    padding_multiple: int = 256
    max_seq_length: int | None = None
    split_group_field: str | None = None
    save_every: int = 50
    eval_every: int = 50

    def __post_init__(self):
        if self.role not in {"consciousness", "subconsciousness"}:
            raise ValueError("role: consciousness или subconsciousness")
        for key in ("seed", "lora_rank", "epochs", "gradient_accumulation",
                    "loss_chunk_tokens", "padding_multiple", "save_every", "eval_every"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == "seed" else 1):
                raise ValueError(f"Недопустимое значение {key}")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction должна быть между 0 и 1")
        if not 0 <= self.lora_dropout < 1 or not 0 <= self.warmup_ratio < 1:
            raise ValueError("Недопустимые dropout/warmup")
        if self.learning_rate <= 0 or self.lora_alpha <= 0 or self.max_grad_norm <= 0 or self.weight_decay < 0:
            raise ValueError("Недопустимые параметры оптимизации")
        if self.max_seq_length is not None and self.max_seq_length < 2:
            raise ValueError("max_seq_length должна быть >= 2 или null")
        if self.split_group_field is not None and (not isinstance(self.split_group_field, str)
                                                   or not self.split_group_field.strip()):
            raise ValueError("split_group_field должен быть непустым именем поля или null")

    def path(self, name):
        path = Path(getattr(self, name)).expanduser()
        return path if path.is_absolute() else ROOT / path

    def as_dict(self):
        return asdict(self)


def load_config(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(raw) - {f.name for f in fields(TrainConfig)}
    if unknown:
        raise ValueError(f"Неизвестные поля конфигурации: {sorted(unknown)}")
    return TrainConfig(**raw)
