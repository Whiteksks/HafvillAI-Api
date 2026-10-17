"""Load models and generate responses in one dedicated thread."""

import gc
import json
import logging
import time

import torch

from hafvillai_runtime.decoding import generate_tokens, repetition_cycle, stabilize_inference_norm
from hafvillai_runtime.config import ModelSpec
from inference.artifact_store import verify_artifacts
from hafvillai_runtime.config import SERIALIZATION
from hafvillai_runtime.serialization import ChatSerializer
from hafvillai_runtime.lora import load_adapter
from hafvillai_runtime.model import assert_kernel_dtypes, load_base

from .schemas import GenerateResponse, InputTooLong, Usage

logger = logging.getLogger(__name__)


class MambaEngine:
    def __init__(self, settings):
        self.settings = settings
        self.models = {}
        self.serializers = {}
        self.steps = {}
        self.hooks = []

    def load(self):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Inference requires an NVIDIA GPU with CUDA and BF16 support.")
        verify_artifacts(self.settings.artifacts_dir)
        config = ModelSpec()
        stabilize_inference_norm()
        roles = ["consciousness"]
        if self.settings.enable_subconsciousness:
            roles.append("subconsciousness")
        from transformers import AutoTokenizer

        for role in roles:
            directory = self.settings.artifacts_dir / role
            metadata = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
            expected = {"id": config.tokenizer_id, "revision": config.tokenizer_revision, "format": SERIALIZATION}
            if metadata.get("role") != role or metadata.get("tokenizer") != expected:
                raise ValueError(f"{role}: роль или токенизатор адаптера не совпадают с SFT")
            tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True)
            serializer = ChatSerializer(tokenizer)
            logger.info("Loading %s, checkpoint %s", role, metadata["step"])
            model, base = load_base(config, offline=True, local_dir=self.settings.artifacts_dir / "base")
            # Retain the model before attaching LoRA so cleanup also handles failures.
            self.models[role] = model
            load_adapter(model, directory, expected_base=base)
            model.requires_grad_(False).eval()
            assert_kernel_dtypes(model)
            self.hooks.append(model.lm_head.register_forward_hook(self._finite_logits))
            self.serializers[role] = serializer
            self.steps[role] = metadata["step"]
            # Exercise real prefill/step paths and warm kernels before reporting ready.
            self.generate(role, [{"role": "user", "content": "Hello."}], 2, "warmup")
        logger.info("Ready: %s; allocated VRAM %.2f GiB", ", ".join(roles), torch.cuda.memory_allocated() / 2**30)

    @staticmethod
    def _finite_logits(module, inputs, output):
        if not torch.isfinite(output).all().item():
            raise RuntimeError("В logits обнаружены NaN/Inf")

    @torch.inference_mode()
    def generate(self, role, messages, max_new_tokens, request_id):
        serializer = self.serializers[role]
        prompt = serializer.prompt(messages)
        if len(prompt) > self.settings.max_input_tokens:
            raise InputTooLong(f"Вход: {len(prompt)} токенов; лимит: {self.settings.max_input_tokens}. Текст не обрезан.")
        ids = torch.tensor([prompt], dtype=torch.long, device="cuda")
        torch.cuda.synchronize()
        started = time.perf_counter()
        # Fresh InferenceParams per request; the penalty does not affect input facts.
        eos = serializer.tokenizer.eos_token_id
        tokens = generate_tokens(self.models[role], ids, max_new_tokens, {serializer.eot, eos},
                                 "greedy", 42, getattr(self.settings, f"{role}_repetition_penalty"),
                                 stop_on_repetition=True)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        if not tokens:
            raise RuntimeError("Модель не сгенерировала токены")
        cycle = repetition_cycle(tokens) if tokens[-1] not in {serializer.eot, eos} else None
        visible_tokens = tokens[:cycle['start']+cycle['period']] if cycle else tokens
        finish = ("eot" if tokens[-1] == serializer.eot else "eos" if tokens[-1] == eos else
                  "repetition" if cycle else "max_new_tokens")
        return GenerateResponse(
            request_id=request_id, role=role, adapter_step=self.steps[role],
            content=serializer.tokenizer.decode(visible_tokens, skip_special_tokens=True),
            finish_reason=finish,
            usage=Usage(prompt_tokens=len(prompt), completion_tokens=len(tokens), total_tokens=len(prompt) + len(tokens)),
            generation_seconds=round(seconds, 4),
        )

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.models.clear()
        self.serializers.clear()
        self.steps.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
