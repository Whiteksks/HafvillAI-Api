"""Single-process, single-GPU SFT with local MIMO autograd and indexing fixes."""

import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import sys
import time

import torch

from .checkpoints import load_checkpoint, save_checkpoint, write_json
from .config import ROOT, load_config
from hafvillai_runtime.config import SERIALIZATION
from .data import collate_one, load_prepared, load_tokenizer
from hafvillai_runtime.lora import adapter_parameters, adapter_state, attach_lora, restore_adapter
from hafvillai_runtime.model import assert_kernel_dtypes, load_base
from .model import batch_loss, forward_hidden


def source_fingerprint():
    from mamba_ssm.modules import mamba3
    expected = ROOT / "mamba-source" / "mamba_ssm"
    if Path(mamba3.__file__).resolve().parent.parent != expected.resolve():
        raise ValueError("Подключи локальный mamba-source через editable install: см. training/setup_brev.sh")
    digest = hashlib.sha256()
    paths = sorted(expected.rglob("*.py")) + sorted((ROOT / "training").glob("*.py")) + sorted((ROOT / "hafvillai_runtime").glob("*.py"))
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def epoch_order(count, seed, epoch):
    order = list(range(count))
    random.Random(seed + epoch).shuffle(order)
    return order


def lr_multiplier(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    fraction = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * fraction))


def check_trainable(model):
    adapters = adapter_parameters(model)
    if not adapters:
        raise ValueError("Нет обучаемых адаптеров")
    for name, p in model.named_parameters():
        if p.requires_grad != (name in adapters):
            raise ValueError(f"Некорректный requires_grad: {name}")
    assert_kernel_dtypes(model)
    return list(adapters.values())


def check_gradients(model):
    for name, p in adapter_parameters(model).items():
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise RuntimeError(f"Отсутствующий или нечисловой градиент: {name}")


@torch.no_grad()
def evaluate(model, rows, config, pad_id):
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    try:
        for index, item in enumerate(rows):
            ids, labels = collate_one(item, pad_id, config.padding_multiple)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, count = batch_loss(model, ids.cuda(), labels.cuda(), config)
            if not torch.isfinite(loss):
                raise RuntimeError("Validation loss нечисловая")
            total_loss += loss.item()
            total_tokens += count
            if (index + 1) % 25 == 0:
                print(f"validation: {index + 1}/{len(rows)}", flush=True)
    finally:
        model.train(was_training)
    return total_loss / total_tokens


def verify_smoke_roundtrip(model, checkpoint_path, probe_ids):
    from safetensors.torch import load_file
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = model.backbone(probe_ids).clone()
    saved = adapter_state(model)
    with torch.no_grad():
        for p in adapter_parameters(model).values():
            p.zero_()
    restore_adapter(model, load_file(str(checkpoint_path / "adapter.safetensors")))
    for name, tensor in adapter_state(model).items():
        torch.testing.assert_close(tensor, saved[name], rtol=0, atol=0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        actual = model.backbone(probe_ids)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    print("SMOKE OK: backward, обновление LoRA, validation, сохранение/загрузка", flush=True)


def run(config, *, offline=False, resume=None, max_steps=None, smoke=False, smoke_long=False):
    prepared, train_rows, valid_rows = load_prepared(config)
    smoke = smoke or smoke_long
    if smoke:
        if resume is not None:
            raise ValueError("Smoke и resume несовместимы")
        config = replace(config, output_dir=str(config.path("output_dir")) + ("-smoke-long" if smoke_long else "-smoke"),
                         epochs=1, gradient_accumulation=1, eval_every=2, save_every=2)
        train_rows = sorted(train_rows, key=lambda item: len(item["input_ids"]), reverse=smoke_long)[:2]
        valid_rows = sorted(valid_rows, key=lambda item: len(item["input_ids"]), reverse=smoke_long)[:2]
        max_steps = 2
    total_steps = math.ceil(len(train_rows) / config.gradient_accumulation) * config.epochs
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("max_steps должна быть положительной")
        total_steps = min(total_steps, max_steps)
    output = config.path("output_dir")
    if resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Каталог уже содержит запуск: {output}. Используй --resume latest или новый --output-dir.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Обучение требует NVIDIA GPU с BF16")
    torch.cuda.set_device(0)
    seed_all(config.seed)
    model, base = load_base(config, offline=offline)
    if prepared["report"]["tokenizer_vocab_size"] > model.lm_head.out_features:
        raise ValueError("Размер токенизатора превышает словарь модели")
    chunk_sizes = {block.mixer.chunk_size for block in model.backbone.layers}
    if any(config.padding_multiple % size for size in chunk_sizes):
        raise ValueError("padding_multiple должна быть кратна всем MIMO chunk_size")
    attach_lora(model, config.lora_rank, config.lora_alpha, config.lora_dropout)
    parameters = check_trainable(model)
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    warmup = math.ceil(total_steps * config.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_multiplier(step, total_steps, warmup))
    manifest = {
        "config": config.as_dict(), "base": base, "prepared": prepared,
        "source_fingerprint": source_fingerprint(), "total_steps": total_steps,
        "tokenizer": {"id": config.tokenizer_id, "revision": config.tokenizer_revision, "format": SERIALIZATION},
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "huggingface-hub", "safetensors", "tilelang", "triton",
                      "apache-tvm-ffi", "nvidia-cutlass-dsl", "quack-kernels")},
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    }
    progress = {"step": 0, "epoch": 0, "next_index": 0, "best_validation_loss": None}
    if resume is not None:
        if resume == "latest":
            resume_path = output / json.loads((output / "latest.json").read_text())["checkpoint"]
        else:
            resume_path = Path(resume).resolve()
        progress = load_checkpoint(resume_path, model, optimizer, scheduler, manifest)
    else:
        output.mkdir(parents=True, exist_ok=True)
        load_tokenizer(config, offline).save_pretrained(output / "tokenizer")
        write_json(output / "run.json", manifest)
    if progress["step"] >= total_steps:
        print(f"Запуск уже завершён: {progress['step']}/{total_steps}")
        return
    pad_id = prepared["report"]["padding_token_id"]
    probe_ids = train_rows[0]["input_ids"][:64].long().unsqueeze(0).cuda()
    initial = adapter_state(model) if smoke else None
    if smoke:
        model.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = forward_hidden(model, probe_ids, gradient_checkpointing=True)
            with torch.no_grad():
                expected = model.backbone(probe_ids)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
        del actual, expected
    model.train()
    print(f"GPU: {torch.cuda.get_device_name(0)}; LoRA: {sum(p.numel() for p in parameters):,} FP32 параметров; "
          f"шагов: {total_steps}; полный текст, microbatch=1", flush=True)
    torch.cuda.reset_peak_memory_stats()
    last_checkpoint = None
    for epoch in range(progress["epoch"], config.epochs):
        order = epoch_order(len(train_rows), config.seed, epoch)
        offset = progress["next_index"] if epoch == progress["epoch"] else 0
        for start in range(offset, len(order), config.gradient_accumulation):
            group = [train_rows[i] for i in order[start:start + config.gradient_accumulation]]
            denominator = sum(item["target_tokens"] for item in group)
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            sum_loss = 0.0
            for item in group:
                ids, labels = collate_one(item, pad_id, config.padding_multiple)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, count = batch_loss(model, ids.cuda(), labels.cuda(), config)
                if count != item["target_tokens"] or not torch.isfinite(loss):
                    raise RuntimeError(f"Некорректная loss/число токенов в исходной строке {item['row']}")
                (loss / denominator).backward()
                sum_loss += loss.item()
            check_gradients(model)
            norm = torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm, error_if_nonfinite=True)
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            progress["step"] += 1
            next_index = start + len(group)
            progress["epoch"] = epoch + 1 if next_index == len(order) else epoch
            progress["next_index"] = 0 if next_index == len(order) else next_index
            metrics = {"step": progress["step"], "epoch": epoch + 1, "loss": sum_loss / denominator,
                       "assistant_tokens": denominator, "grad_norm": float(norm), "lr": lr,
                       "seconds": round(time.perf_counter() - started, 2),
                       "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                       "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
            final = progress["step"] == total_steps
            improved = False
            due_eval = progress["step"] % config.eval_every == 0 or final
            if due_eval:
                value = evaluate(model, valid_rows, config, pad_id)
                metrics["validation_loss"] = value
                improved = progress["best_validation_loss"] is None or value < progress["best_validation_loss"]
                if improved:
                    progress["best_validation_loss"] = value
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics) + "\n")
            if progress["step"] == 1 or progress["step"] % config.save_every == 0 or due_eval:
                last_checkpoint = save_checkpoint(output, model, optimizer, scheduler, progress, manifest)
                if improved:
                    write_json(output / "best.json", {"checkpoint": last_checkpoint.name,
                                                       "validation_loss": progress["best_validation_loss"]})
            if final:
                if smoke:
                    current = adapter_state(model)
                    for suffix in (".lora_A", ".lora_B"):
                        if not any(not torch.equal(current[k], initial[k]) for k in current if k.endswith(suffix)):
                            raise RuntimeError(f"Smoke: параметры {suffix} не обновились")
                    check_trainable(model)
                    verify_smoke_roundtrip(model, last_checkpoint, probe_ids)
                write_json(output / "completed.json", {"step": progress["step"], "checkpoint": last_checkpoint.name})
                print(f"Готово: {last_checkpoint}", flush=True)
                return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", help="latest или путь к checkpoint")
    parser.add_argument("--max-steps", type=int, help="Ограничить число обновлений; входит в идентичность запуска")
    parser.add_argument("--output-dir", help="Новый каталог запуска")
    parser.add_argument("--smoke", action="store_true", help="2 шага на коротких реальных примерах в отдельном каталоге")
    parser.add_argument("--smoke-long", action="store_true", help="Та же проверка на самых длинных записях, отдельный каталог")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config = replace(config, output_dir=str(Path(args.output_dir).resolve()))
    run(config, offline=args.offline, resume=args.resume, max_steps=args.max_steps,
        smoke=args.smoke, smoke_long=args.smoke_long)


if __name__ == "__main__":
    main()
