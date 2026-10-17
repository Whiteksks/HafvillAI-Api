"""Serialize unchanged messages as Llama-3 headers with assistant-only labels.

Each dialogue stays a separate sequence. Summaries share their source dialogue's
split; embedded dialogue text matches the Dataset Studio export exactly.
"""

import hashlib
import json
import math
from pathlib import Path
import random
import tempfile

import torch

from hafvillai_runtime.files import file_sha256
from hafvillai_runtime.config import SERIALIZATION
from hafvillai_runtime.serialization import ChatSerializer, IGNORE_INDEX


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_rows(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                messages = row["messages"]
                if not isinstance(messages, list) or not messages:
                    raise ValueError("нет messages")
                for m in messages:
                    if m["role"] not in {"system", "user", "assistant"} or not isinstance(m["content"], str):
                        raise ValueError("неверный формат сообщения")
                if not any(m["role"] == "assistant" and m["content"] for m in messages):
                    raise ValueError("нет целевого ответа assistant")
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}, строка {number}: {exc}") from exc
            rows.append(row)
    if not rows:
        raise ValueError(f"Пустой датасет: {path}")
    return rows


def dialogue_text(row):
    # Match dataset-studio/summaries.py:dialogue_text exactly.
    return "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in row["messages"])


def metadata_group(row, field):
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Нет непустого строкового поля split: {field}")
    return text_sha256(value)


def validation_groups(source_rows, seed, fraction, field=None):
    groups = sorted({metadata_group(row, field) if field else text_sha256(dialogue_text(row))
                     for row in source_rows})
    if len(groups) < 2:
        raise ValueError("Для split нужны минимум два разных диалога")
    random.Random(seed).shuffle(groups)
    count = min(len(groups) - 1, max(1, math.ceil(len(groups) * fraction)))
    return set(groups[:count]), set(groups)


def group_id(row, role, field=None):
    if field:
        return metadata_group(row, field)
    if role == "consciousness":
        return text_sha256(dialogue_text(row))
    messages = row["messages"]
    if [m["role"] for m in messages] != ["system", "user", "assistant"]:
        raise ValueError("Summary ожидает system, user (полный диалог), assistant (сводка)")
    return text_sha256(messages[1]["content"])


def preparation_signature(config):
    result = {
        "format": SERIALIZATION,
        "role": config.role,
        "dataset_sha256": file_sha256(config.path("dataset")),
        "source_sha256": file_sha256(config.path("source_dataset")),
        "tokenizer_id": config.tokenizer_id,
        "tokenizer_revision": config.tokenizer_revision,
        "seed": config.seed,
        "validation_fraction": config.validation_fraction,
        "max_seq_length": config.max_seq_length,
    }
    if config.split_group_field:
        result["split_group_field"] = config.split_group_field
    return result


def load_tokenizer(config, offline=False):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    snapshot = snapshot_download(config.tokenizer_id, revision=config.tokenizer_revision,
                                 allow_patterns=["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
                                 local_files_only=offline)
    # A local path also prevents an implicit model_info request in Transformers 4.57.6.
    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True)


def prepare(config, *, offline=False):
    target = config.path("prepared_dir") / config.role
    signature = preparation_signature(config)
    if target.exists():
        manifest, _, _ = load_prepared(config)
        return manifest["report"]
    source = read_rows(config.path("source_dataset"))
    valid_groups, all_groups = validation_groups(source, config.seed, config.validation_fraction,
                                                config.split_group_field)
    rows = read_rows(config.path("dataset"))
    serializer = ChatSerializer(load_tokenizer(config, offline))
    train, validation, lengths = [], [], []
    for index, row in enumerate(rows):
        group = group_id(row, config.role, config.split_group_field)
        if group not in all_groups:
            raise ValueError(f"Запись {index + 1} не найдена в исходных группах; нельзя согласовать split")
        ids, labels = serializer.encode(row["messages"])
        size = len(ids)
        if config.max_seq_length is not None and size > config.max_seq_length:
            raise ValueError(f"Запись {index + 1}: {size} > max_seq_length={config.max_seq_length}. "
                             "Исходный текст не обрезан. Увеличь лимит или укажи null.")
        count = int((labels[1:] != IGNORE_INDEX).sum())
        if not count:
            raise ValueError(f"Запись {index + 1}: нет целевых токенов")
        item = {"input_ids": ids, "labels": labels, "group": group, "row": index + 1,
                "target_tokens": count}
        (validation if group in valid_groups else train).append(item)
        lengths.append(size)
        if (index + 1) % 250 == 0:
            print(f"{config.role}: токенизировано {index + 1}/{len(rows)}", flush=True)
    if not train or not validation:
        raise ValueError("Получился пустой train или validation split")
    ordered = sorted(lengths)
    report = {"records": len(rows), "train": len(train), "validation": len(validation),
              "min_tokens": ordered[0], "median_tokens": ordered[len(ordered) // 2],
              "p95_tokens": ordered[min(len(ordered)-1, int(len(ordered) * .95))],
              "max_tokens": ordered[-1], "total_tokens": sum(lengths),
              "assistant_tokens": sum(item["target_tokens"] for item in train + validation),
              "beyond_pretraining_2048": sum(n > 2048 for n in lengths),
              "padding_token_id": serializer.eot, "tokenizer_vocab_size": len(serializer.tokenizer)}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=f".{config.role}-") as tmp:
        directory = Path(tmp) / "ready"
        directory.mkdir()
        torch.save(train, directory / "train.pt")
        torch.save(validation, directory / "validation.pt")
        manifest = {"signature": signature, "report": report,
                    "artifact_sha256": {name: file_sha256(directory / name)
                                        for name in ("train.pt", "validation.pt")}}
        (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        directory.rename(target)
    return report


def load_prepared(config):
    directory = config.path("prepared_dir") / config.role
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["signature"] != preparation_signature(config):
        raise ValueError("Данные/настройки подготовки изменились. Выбери новый prepared_dir и выполни prepare.")
    for name, digest in manifest["artifact_sha256"].items():
        if file_sha256(directory / name) != digest:
            raise ValueError(f"Повреждён подготовленный файл {name}")
    train = torch.load(directory / "train.pt", map_location="cpu", weights_only=True)
    validation = torch.load(directory / "validation.pt", map_location="cpu", weights_only=True)
    return manifest, train, validation


def collate_one(item, pad_id, multiple):
    """Right-pad only; Mamba does not use attention_mask."""
    size = len(item["input_ids"])
    padded = math.ceil(size / multiple) * multiple
    ids = torch.full((1, padded), pad_id, dtype=torch.long)
    labels = torch.full((1, padded), IGNORE_INDEX, dtype=torch.long)
    ids[0, :size] = item["input_ids"]
    labels[0, :size] = item["labels"]
    return ids, labels
