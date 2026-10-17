"""Pinned model identity shared by training and inference."""

from dataclasses import dataclass


SERIALIZATION = "llama3-headers-v1"


@dataclass(frozen=True, kw_only=True)
class ModelSpec:
    model_id: str = "state-spaces/mamba3-mimo-1.5b"
    model_revision: str = "bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec"
    model_config_sha256: str = "71e804ad2a077191e45f5775e373df164e28df212377c04625460d60981f1ad0"
    model_weights_sha256: str = "139c09e9728d1f10ac7f4354d2ae29aaca41851a954aedd8f6728c05aeb6c744"
    tokenizer_id: str = "meta-llama/Llama-3.1-8B"
    tokenizer_revision: str = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
