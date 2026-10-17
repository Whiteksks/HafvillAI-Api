from dataclasses import dataclass
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INFERENCE_ROOT = Path(__file__).resolve().parents[1]


def env_bool(name, default):
    raw = os.getenv(name, str(int(default))).lower()
    if raw not in {"0", "1", "false", "true"}:
        raise ValueError(f"{name}: ожидается 0 или 1")
    return raw in {"1", "true"}


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8000
    enable_subconsciousness: bool = True
    max_input_tokens: int = 1536
    max_new_tokens: int = 512
    max_queue_size: int = 4
    max_body_bytes: int = 1048576
    api_key: str = ""
    artifacts_dir: Path = INFERENCE_ROOT / "artifacts"
    consciousness_repetition_penalty: float = 1.0
    subconsciousness_repetition_penalty: float = 1.2

    def __post_init__(self):
        if self.host not in {"127.0.0.1", "localhost", "::1"} and not self.api_key.strip():
            raise ValueError("HAFVILLAI_API_KEY is required when binding to a non-loopback host")
        for name in ("port", "max_input_tokens", "max_new_tokens", "max_queue_size", "max_body_bytes"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} должен быть положительным")
        if self.port > 65535:
            raise ValueError("port должен быть <= 65535")
        if self.max_input_tokens + self.max_new_tokens > 2048:
            raise ValueError("max_input_tokens + max_new_tokens должны быть <= 2048 для CoSER")
        for role in ("consciousness", "subconsciousness"):
            value = getattr(self, f"{role}_repetition_penalty")
            if not math.isfinite(value) or value < 1:
                raise ValueError(f"{role}_repetition_penalty должен быть конечным и >= 1")

    @classmethod
    def from_env(cls):
        def path(name):
            value = os.getenv(name)
            if not value:
                return INFERENCE_ROOT / "artifacts"
            result = Path(value).expanduser()
            return result if result.is_absolute() else ROOT / result

        return cls(
            host=os.getenv("HAFVILLAI_HOST", "127.0.0.1"),
            port=int(os.getenv("HAFVILLAI_PORT", "8000")),
            enable_subconsciousness=env_bool("HAFVILLAI_ENABLE_SUBCONSCIOUSNESS", True),
            max_input_tokens=int(os.getenv("HAFVILLAI_MAX_INPUT_TOKENS", "1536")),
            max_new_tokens=int(os.getenv("HAFVILLAI_MAX_NEW_TOKENS", "512")),
            max_queue_size=int(os.getenv("HAFVILLAI_MAX_QUEUE_SIZE", "4")),
            max_body_bytes=int(os.getenv("HAFVILLAI_MAX_BODY_BYTES", "1048576")),
            api_key=os.getenv("HAFVILLAI_API_KEY", ""),
            artifacts_dir=path("HAFVILLAI_ARTIFACTS_DIR"),
            consciousness_repetition_penalty=float(os.getenv("HAFVILLAI_CONSCIOUSNESS_REPETITION_PENALTY", "1")),
            subconsciousness_repetition_penalty=float(os.getenv("HAFVILLAI_SUBCONSCIOUSNESS_REPETITION_PENALTY", "1.2")),
        )
