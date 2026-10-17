"""Reproducible base/LoRA comparison with a shared 2,048-token context limit."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

import torch
from torch.nn import functional as F

from training.config import ROOT, load_config
from hafvillai_runtime.config import SERIALIZATION
from hafvillai_runtime.serialization import ChatSerializer, IGNORE_INDEX
from hafvillai_runtime.files import file_sha256
from training.data import load_prepared, load_tokenizer, read_rows
from hafvillai_runtime.lora import load_adapter
from hafvillai_runtime.model import assert_kernel_dtypes, load_base

from hafvillai_runtime.decoding import generate_tokens, repetition_cycle, stabilize_inference_norm
from .comparison_metrics import TAGS, output_budget, tagged_spans, text_metrics, token_kinds, validate_cases

def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def select_cases(args):
    if args.suite:
        path = args.cases_file or Path(__file__).parent / "examples/coser_checks.json"
        cases = json.loads(path.read_text(encoding="utf-8"))[args.role]
    elif args.messages_file:
        cases = [{"id": "custom", "format": "chat",
                  "messages": json.loads(args.messages_file.read_text(encoding="utf-8"))}]
    else:
        prompt = args.prompt or "Hi. What is your name?"
        if args.raw:
            if args.system:
                raise ValueError("--system нельзя использовать с --raw")
            cases = [{"id": "custom", "format": "raw", "prompt": prompt}]
        else:
            messages = ([{"role": "system", "content": args.system}] if args.system else [])
            cases = [{"id": "custom", "format": "chat", "messages": messages + [{"role": "user", "content": prompt}]}]
    validate_cases(cases)
    if args.case:
        cases = [case for case in cases if case["id"] in args.case]
        if {case["id"] for case in cases} != set(args.case):
            raise ValueError("Не все --case найдены в наборе")
    return cases


def encode_case(case, serializer):
    if case["format"] == "chat":
        return serializer.prompt(case["messages"])
    return serializer.tokenizer.encode(case["prompt"], add_special_tokens=True)


def finite_logits(module, inputs, output):
    if not torch.isfinite(output).all().item():
        raise RuntimeError("В logits обнаружены NaN/Inf")




@torch.inference_mode()
def verify_generation(model, ids, generated, eot):
    """Check decoding and full forward against cached teacher forcing."""
    from mamba_ssm.utils.generation import InferenceParams
    count = min(32, len(generated))
    official = model.generate(ids, max_length=ids.shape[1] + count, top_k=1,
                              cg=False, eos_token_id=eot)[0, ids.shape[1]:].tolist()
    if official != generated[:count]:
        raise RuntimeError("Greedy-префикс нашего декодера отличается от штатного generate")
    repeated = generate_tokens(model, ids, count, {eot}, "greedy", 42)
    if repeated != generated[:count]:
        raise RuntimeError("Новый кеш изменил greedy-префикс")

    count = min(64, len(generated))
    suffix = torch.tensor([generated[:count - 1]], dtype=torch.long, device=ids.device)
    sequence = torch.cat([ids, suffix], dim=1)
    full = model(sequence, num_last_tokens=count).logits[0].float()
    state = InferenceParams(max_seqlen=sequence.shape[1], max_batch_size=1)
    cached = []
    current = ids
    for index in range(count):
        logits = model(current, inference_params=state, num_last_tokens=1).logits[0, -1].float()
        cached.append(logits)
        state.seqlen_offset += current.shape[1]
        current = torch.tensor([[generated[index]]], dtype=torch.long, device=ids.device)
    cached = torch.stack(cached)
    relative_l2 = float((full - cached).norm() / full.norm().clamp_min(1e-8))
    if relative_l2 > 0.05:
        raise RuntimeError(f"Полный forward и cached step расходятся: relative L2={relative_l2:.6f}")
    return {"official_greedy_prefix_equal": True, "fresh_cache_prefix_equal": True,
            "teacher_forcing_tokens": count, "full_cached_relative_l2": relative_l2,
            "full_cached_top1_agreement": float((full.argmax(-1) == cached.argmax(-1)).float().mean())}


def body_encoding(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return encoded["input_ids"], token_kinds(text, encoded["offset_mapping"])


def audit_dataset(config, tokenizer):
    rows = read_rows(config.path("dataset"))
    counts = Counter()
    body_chars = 0
    target_chars = 0
    for row in rows:
        for message in row["messages"]:
            if message["role"] != "assistant":
                continue
            text = message["content"]
            _, kinds = body_encoding(tokenizer, text)
            counts.update(kinds)
            counts["eot"] += 1
            counts["assistant_messages"] += 1
            target_chars += len(text)
            body_chars += sum(end - start for start, end in tagged_spans(text, "system_thinking", body=True))
    total = sum(counts[kind] for kind in (*TAGS, "other", "eot"))
    return {"role": config.role, "dataset": str(config.path("dataset")),
            "dataset_sha256": file_sha256(config.path("dataset")), "records": len(rows),
            "assistant_messages": counts["assistant_messages"], "assistant_target_tokens": total,
            "categories": {kind: {"tokens": counts[kind], "fraction": counts[kind] / total}
                           for kind in (*TAGS, "other", "eot")},
            "system_thinking_body_character_fraction": body_chars / max(1, target_chars),
            "method": "offset overlap; system_thinking > role_thinking > role_action; delimiters included; EOT separate; other is not necessarily only spoken text"}


def target_categories(row, serializer):
    categories = ["ignored"]
    for message in row["messages"]:
        categories.extend(["ignored"] * len(serializer.headers[message["role"]]))
        body, kinds = body_encoding(serializer.tokenizer, message["content"])
        categories.extend(kinds if message["role"] == "assistant" else ["ignored"] * len(body))
        categories.append("eot" if message["role"] == "assistant" else "ignored")
    return categories


def heldout_items(config, count, max_length):
    manifest, _, validation = load_prepared(config)
    eligible = [item for item in validation if len(item["input_ids"]) <= max_length]
    # Fixed source-row order; selection does not depend on model outputs.
    chosen = sorted(eligible, key=lambda item: item["row"])[:count]
    if not chosen:
        raise ValueError("Нет validation-примеров, помещающихся в общий лимит")
    rows = read_rows(config.path("dataset"))
    return {"validation_total": len(validation), "eligible": len(eligible),
            "selected_rows_one_based": [item["row"] for item in chosen],
            "dataset_sha256": manifest["signature"]["dataset_sha256"]}, [(item, rows[item["row"] - 1]) for item in chosen]


@torch.inference_mode()
def evaluate_heldout(model, serializer, selected):
    sums, counts = defaultdict(float), Counter()
    per_row = []
    for item, row in selected:
        expected_ids, expected_labels = serializer.encode(row["messages"])
        if not torch.equal(expected_ids, item["input_ids"]) or not torch.equal(expected_labels, item["labels"]):
            raise ValueError("Сериализация validation не совпадает с подготовленным SFT")
        categories = target_categories(row, serializer)
        if len(categories) != len(expected_ids):
            raise ValueError("Категории токенов не совпали с сериализацией")
        ids = item["input_ids"].long().unsqueeze(0).cuda()
        labels = item["labels"].long().cuda()
        hidden = model.backbone(ids)[0]
        positions = (labels[1:] != IGNORE_INDEX).nonzero().flatten()
        local_sums, local_counts = defaultdict(float), Counter()
        for start in range(0, len(positions), 128):
            indices = positions[start:start + 128]
            logits = model.lm_head(hidden[indices]).float()
            losses = F.cross_entropy(logits, labels[indices + 1], reduction="none").cpu().tolist()
            for position, loss in zip(indices.cpu().tolist(), losses):
                category = categories[position + 1]
                local_sums[category] += loss
                local_counts[category] += 1
        for category in local_counts:
            sums[category] += local_sums[category]
            counts[category] += local_counts[category]
        per_row.append({"row": item["row"], "tokens": len(expected_ids),
                        "loss": sum(local_sums.values()) / sum(local_counts.values())})
        del hidden, ids, labels
    return {"overall_loss": sum(sums.values()) / sum(counts.values()),
            "categories": {key: {"tokens": counts[key], "loss": sums[key] / counts[key]} for key in counts},
            "rows": per_row, "note": "Teacher forcing on complete validation rows <= max_length; not free-generation quality"}


def write_summary(output, records):
    lines = ["# Автоматические показатели сравнения", "", "Это метрики повторов и формата, а не оценка смысла.", "",
             "| Модель | Режим | Тест | Токены | Остановка | Повтор 4-грамм | Планирование | Видимых символов |",
             "|---|---|---|---:|---|---:|---|---:|"]
    for item in records:
        metrics = item["metrics"]
        lines.append(f"| {item['mode']} | {item['decoding']} seed={item['seed']} | {item['case_id']} | "
                     f"{item['generated_tokens']} | {item['stop_reason']} | "
                     f"{metrics['all_text']['duplicate_4gram_fraction']:.3f} | "
                     f"{metrics['planning_marker_present']} | {metrics['visible_chars']} |")
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["consciousness", "subconsciousness"], default="consciousness")
    parser.add_argument("--config", type=Path, help="Отдельная конфигурация данных для аудита/validation")
    parser.add_argument("--mode", choices=["base", "lora", "both"], default="both")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--prompt")
    inputs.add_argument("--messages-file", type=Path)
    inputs.add_argument("--suite", action="store_true")
    parser.add_argument("--system")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--cases-file", type=Path)
    parser.add_argument("--case", action="append")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--max-length", type=positive_int, default=2048)
    parser.add_argument("--max-new-tokens", type=positive_int)
    parser.add_argument("--decode", choices=["greedy", "sample", "both"], default="greedy")
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--stop-on-repetition", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--stop", choices=["both", "eot", "eos"], default="both")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--heldout-eval", type=int, default=0)
    parser.add_argument("--online", action="store_true", help="Разрешить получение базы/токенизатора из HF")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.repetition_penalty) or args.repetition_penalty < 1.0:
        parser.error("repetition_penalty must be finite and >= 1")
    if args.verify and (args.repetition_penalty != 1.0 or args.stop_on_repetition):
        parser.error("--verify requires repetition_penalty=1; verify unmodified decoding separately")
    if args.max_length > 2048 or args.heldout_eval < 0:
        parser.error("max_length <= 2048, heldout_eval >= 0")
    if args.raw and (args.suite or args.messages_file):
        parser.error("--raw предназначен для --prompt")
    if args.system and (args.suite or args.messages_file):
        parser.error("--system предназначен для --prompt")
    if args.cases_file and not args.suite:
        parser.error("--cases-file требует --suite")
    cases = select_cases(args)
    config_name = "coser_sft.json" if args.role == "consciousness" else "coser_subconsciousness.json"
    config = load_config(args.config or ROOT / "training/configs" / config_name)
    if config.role != args.role:
        parser.error("Роль --config должна совпадать с --role")
    folder = ROOT / "outputs/brev-coser" / args.role
    directory = args.adapter.resolve() if args.adapter else None
    metadata = None
    if args.mode != "base" and not args.audit_only:
        if directory is None:
            best = json.loads((folder / "best.json").read_text(encoding="utf-8"))
            directory = folder / best["checkpoint"]
        metadata = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
        expected = {"id": config.tokenizer_id, "revision": config.tokenizer_revision, "format": SERIALIZATION}
        if metadata.get("role") != args.role or metadata.get("tokenizer") != expected:
            raise ValueError("Роль или токенизатор адаптера не совпали с SFT")
    from transformers import AutoTokenizer
    tokenizer_dir = (directory.parent if directory else folder) / "tokenizer"
    tokenizer = (AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
                 if tokenizer_dir.exists() else load_tokenizer(config, offline=not args.online))
    serializer = ChatSerializer(tokenizer)
    encoded = [(case, encode_case(case, serializer)) for case in cases]
    max_new = args.max_new_tokens if args.max_new_tokens is not None else (256 if args.suite else None)
    for _, prompt in encoded:
        output_budget(len(prompt), args.max_length, max_new)

    if args.output_dir:
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        root = ROOT / "outputs/comparison"
        root.mkdir(parents=True, exist_ok=True)
        prefix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + args.role + "-"
        output = Path(tempfile.mkdtemp(prefix=prefix, dir=root))
    print(f"Результаты: {output}", flush=True)
    run = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "role": args.role,
           "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
           "script_sha256": file_sha256(Path(__file__)),
           "decoding_sha256": file_sha256(Path(generate_tokens.__code__.co_filename)),
           "metrics_sha256": file_sha256(Path(__file__).with_name("comparison_metrics.py")),
           "cases": cases, "status": "started"}
    save_json(output / "run.json", run)
    if args.audit or args.audit_only:
        print("Аудит целевых токенов датасета...", flush=True)
        audit = audit_dataset(config, tokenizer)
        save_json(output / "dataset_audit.json", audit)
        print(json.dumps(audit["categories"], ensure_ascii=False), flush=True)
    if args.audit_only:
        run["status"] = "completed"
        save_json(output / "run.json", run)
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Нужен NVIDIA GPU с CUDA/BF16; запусти в рабочем WSL-терминале")

    selected = []
    if args.heldout_eval:
        selection, selected = heldout_items(config, args.heldout_eval, args.max_length)
        run["heldout_selection"] = selection
    print(f"GPU: {torch.cuda.get_device_name(0)}; загрузка исходной базы...", flush=True)
    stabilize_inference_norm()
    run['rms_norm_num_warps'] = 4
    model, base = load_base(config, offline=not args.online)
    run.update(base=base, gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
               adapter=str(directory) if directory else None,
               adapter_sha256=file_sha256(directory / "adapter.safetensors") if metadata else None,
               adapter_step=metadata["step"] if metadata else None,
               tokenizer={"id": config.tokenizer_id, "revision": config.tokenizer_revision, "format": SERIALIZATION})
    save_json(output / "run.json", run)
    hook = model.lm_head.register_forward_hook(finite_logits)
    records = []
    verification = {}
    teacher = {}
    try:
        modes = ["base", "lora"] if args.mode == "both" else [args.mode]
        for mode in modes:
            if mode == "lora":
                load_adapter(model, directory, expected_base=base)
            model.requires_grad_(False).eval()
            assert_kernel_dtypes(model)
            if selected:
                print(f"{mode}: teacher forcing на {len(selected)} validation-записях...", flush=True)
                teacher[mode] = evaluate_heldout(model, serializer, selected)
                save_json(output / "teacher_forcing.json", teacher)
            decodings = ["greedy", "sample"] if args.decode == "both" else [args.decode]
            for decoding in decodings:
                seeds = [None] if decoding == "greedy" else args.seeds
                for seed in seeds:
                    for case, prompt in encoded:
                        budget = output_budget(len(prompt), args.max_length, max_new)
                        # Use the same end markers for both formats and model variants:
                        # the base may emit EOS while an adapter emits EOT.
                        eos = tokenizer.eos_token_id
                        if not isinstance(eos, int):
                            raise ValueError("Ожидался целочисленный EOS токенизатора")
                        stops = ({serializer.eot, eos} if args.stop == "both" else
                                 {serializer.eot} if args.stop == "eot" else {eos})
                        ids = torch.tensor([prompt], dtype=torch.long, device="cuda")
                        print(f"START {mode}/{decoding}/{case['id']} seed={seed}, вход={len(prompt)}, выход<={budget}", flush=True)
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        tokens = generate_tokens(model, ids, budget, stops, decoding, seed or 0,
                                                 args.repetition_penalty, args.stop_on_repetition)
                        torch.cuda.synchronize()
                        seconds = time.perf_counter() - started
                        raw = tokenizer.decode(tokens, skip_special_tokens=False)
                        metrics = text_metrics(raw, case.get("expected_mentions", []))
                        stop = ("eot" if tokens[-1] == serializer.eot else "eos" if tokens[-1] == eos else
                                "repetition" if args.stop_on_repetition and repetition_cycle(tokens) else "budget")
                        stem = f"{mode}-{decoding}-seed{seed}-{case['id']}"
                        (output / f"{stem}.txt").write_text(raw + "\n", encoding="utf-8")
                        record = {"case_id": case["id"], "format": case["format"], "mode": mode,
                                  "decoding": decoding, "seed": seed, "stop_ids": sorted(stops),
                                  "prompt_tokens": len(prompt), "input_ids_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                                  "output_budget": budget, "generated_tokens": len(tokens), "generated_token_ids": tokens,
                                  "stop_reason": stop, "seconds": seconds,
                                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                                  "metrics": metrics, "text": raw, "text_file": f"{stem}.txt"}
                        records.append(record)
                        with (output / "generations.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        write_summary(output, records)
                        print(f"DONE {mode}/{decoding}/{case['id']}: {len(tokens)} токенов, {seconds:.2f}с, stop={stop}, "
                              f"repeat4={metrics['all_text']['duplicate_4gram_fraction']:.3f}, visible={metrics['visible_chars']}", flush=True)
                        if not args.suite:
                            print(raw[:1500] + ("\n[Полный ответ сохранён в .txt]" if len(raw) > 1500 else ""), flush=True)
                        if args.verify and decoding == "greedy" and mode not in verification:
                            verification[mode] = verify_generation(model, ids, tokens, serializer.eot)
                            save_json(output / "verification.json", verification)
                            print(f"VERIFY {mode}: {verification[mode]}", flush=True)
        run["status"] = "completed"
    finally:
        save_json(output / "run.json", run)
        hook.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    print(f"OK: {len(records)} генераций. Отчёт: {output / 'SUMMARY.md'}", flush=True)


if __name__ == "__main__":
    main()
