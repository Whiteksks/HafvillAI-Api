"""Full book-selected CoSER SFT corpus. No training or model generation here."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import tempfile

import torch

from coser_catalog import POLICY as BOOK_POLICY
from coser_data import bounded_examples, character_exchanges, clean_utterances, make_system
from coser_download import DEFAULT_ROOT, REVISION
from data_utils import write_jsonl
from data_utils import atomic_json, normalize, sha256, source_book_key
from training.config import ROOT, load_config
from hafvillai_runtime.serialization import ChatSerializer, IGNORE_INDEX
from training.data import group_id, load_prepared, load_tokenizer, read_rows, text_sha256, validation_groups

CONFIG = ROOT / "training/configs/coser_sft.json"
POLICY = "coser-thoughtless-book-selected-sft-v1"
MAX_REPLY = 512


def fingerprint(utterances):
    # Speaker aliases and generated actions do not turn the same spoken scene
    # into an independent conversation. Very short scenes also keep actions.
    speakers = {}
    parts = []
    for item in utterances:
        speaker = speakers.setdefault(item["character"], len(speakers))
        content = item["speech"] or item["content"]
        parts.append(f"{speaker}:{normalize(content)}")
    return text_sha256("\n".join(parts))


def corrupt_scene(utterances):
    """Reject obvious generated loops; ordinary short repeated replies are valid."""
    for item in utterances:
        words = normalize(item["speech"]).split()
        if len(words) < 48:
            continue
        for period in range(1, 9):
            for start in range(len(words) - period * 6 + 1):
                unit = words[start:start + period]
                if period * 6 >= 24 and words[start:start + period * 6] == unit * 6:
                    return True
    return False


def statistics(rows, serializer):
    lengths, systems, replies = [], [], []
    targets, input_tokens = 0, 0
    source_scenes, speakers, target_texts = set(), set(), set()
    for row in rows:
        ids, labels = serializer.encode(row["messages"])
        lengths.append(len(ids))
        targets += int((labels[1:] != IGNORE_INDEX).sum())
        input_tokens += len(ids)
        systems.append(len(serializer.tokenizer.encode(row["messages"][0]["content"], add_special_tokens=False)))
        source = row["source"]
        source_scenes.add((source["book"], source["plot_index"], source["conversation_index"]))
        speakers.add((source["book"], source["character"]))
        for message in row["messages"]:
            if message["role"] == "assistant":
                replies.append(len(serializer.tokenizer.encode(message["content"], add_special_tokens=False)) + 1)
                target_texts.add(text_sha256(message["content"]))
    def distribution(values):
        values = sorted(values)
        return {"min": values[0], "median": values[len(values) // 2],
                "p95": values[min(len(values) - 1, math.ceil(len(values) * .95) - 1)], "max": values[-1]}
    return {"examples": len(rows), "source_scenes": len(source_scenes),
            "books": len({r["split_group"] for r in rows}), "characters": len(speakers),
            "assistant_turns": len(replies), "unique_assistant_texts": len(target_texts),
            "total_tokens": input_tokens, "assistant_target_tokens": targets,
            "sequence_tokens": distribution(lengths), "system_tokens": distribution(systems),
            "reply_tokens_including_eot": distribution(replies)}


def export_prompts(rows, serializer, split, directory, limit=16):
    buckets = defaultdict(list)
    for row in sorted(rows, key=lambda r: text_sha256(r["example_id"])):
        buckets[row["split_group"]].append(row)
    cases, references, seen = [], [], set()
    while buckets and len(cases) < limit:
        for book in sorted(list(buckets)):
            row = buckets[book].pop()
            if not buckets[book]:
                del buckets[book]
            source = row["source"]
            scene = (source["book"], source["plot_index"], source["conversation_index"])
            if scene in seen:
                continue
            index = len(row["messages"]) - 1
            while index >= 2 and len(serializer.prompt(row["messages"][:index])) + MAX_REPLY > 2048:
                index -= 2
            if index < 2:
                continue
            seen.add(scene)
            name = f"coser_{split}_{len(cases) + 1:02d}"
            cases.append({"id": name, "format": "chat", "messages": row["messages"][:index]})
            references.append({"id": name, "example_id": row["example_id"], "split": split,
                               "assistant_message_index": index, "reference": row["messages"][index]["content"],
                               "source": source})
            if len(cases) >= limit:
                break
    if not cases:
        raise ValueError(f"No usable {split} prompts")
    atomic_json(directory / f"{split}_cases.json", {"consciousness": cases, "subconsciousness": []})
    atomic_json(directory / f"{split}_references.json", references)


def build(root=DEFAULT_ROOT, config_path=CONFIG):
    root = Path(root)
    config = load_config(config_path)
    destination = config.path("dataset").parent
    if destination.exists():
        raise FileExistsError(f"Export already exists: {destination}. Use check-sft or a new config/destination.")
    if (config.path("source_dataset") != config.path("dataset") or config.max_seq_length != 2048
            or config.split_group_field != "split_group"):
        raise ValueError("SFT requires one train/validation source, explicit book groups and context 2048")
    catalog = json.loads((root / "book_catalog.json").read_text())
    downloaded = json.loads((root / "download_manifest.json").read_text())
    if catalog["policy"] != BOOK_POLICY or catalog["revision"] != REVISION:
        raise ValueError("Book selection policy/revision mismatch")
    if sha256(root / "download_manifest.json") != catalog["download_manifest_sha256"]:
        raise ValueError("Download manifest changed")
    tokenizer = load_tokenizer(config, offline=True)
    serializer = ChatSerializer(tokenizer)
    ood = {source_book_key(s.strip()) for s in (root / "raw/test/ood_books_for_coser_models.txt").read_text().splitlines() if s.strip()}
    selected = [b for b in catalog["books"] if b["status"] == "selected"]
    selected.sort(key=lambda b: (source_book_key(b["book"]) not in ood, b["book"]))
    if not any(source_book_key(b["book"]) in ood for b in selected):
        raise ValueError("No officially OOD books available for a separate test set")
    rows, seen, exclusions, book_report = [], set(), Counter(), []
    for n, book in enumerate(selected, 1):
        title, name = book["book"], book["file"]
        if sha256(root / "raw" / name) != downloaded["files"][name]["sha256"]:
            raise ValueError(f"Changed source file: {name}")
        data = json.loads((root / "raw" / name).read_text())
        is_test = source_book_key(title) in ood
        split_index = data.get("split_plot_index")
        if not isinstance(split_index, int) or not 0 <= split_index <= len(data["plots"]):
            raise ValueError(f"Invalid official split: {title}")
        count, scenes = len(rows), 0
        for i_p, plot in enumerate(data["plots"]):
            if not is_test and i_p >= split_index:
                exclusions["official_id_heldout_plot"] += 1
                continue
            for i_c, conversation in enumerate(plot.get("conversation", [])):
                try:
                    utterances = clean_utterances(conversation)
                except (ValueError, TypeError, AttributeError):
                    exclusions["malformed_or_non_dialogue_scene"] += 1
                    continue
                if corrupt_scene(utterances):
                    exclusions["obvious_source_loop"] += 1
                    continue
                key = fingerprint(utterances)
                if key in seen:
                    exclusions["duplicate_spoken_scene"] += 1
                    continue
                seen.add(key)
                scenes += 1
                for character in sorted({u["character"] for u in utterances}):
                    try:
                        system = make_system(title, character, plot, conversation)
                    except ValueError:
                        exclusions["invalid_system_context"] += 1
                        continue
                    if re_control(system):
                        exclusions["system_contains_control_or_thought_markup"] += 1
                        continue
                    examples, skipped = bounded_examples(system, character_exchanges(utterances, character),
                                                         serializer, max_length=2048, max_reply=MAX_REPLY)
                    exclusions.update(skipped)
                    for i_w, example in enumerate(examples):
                        messages = example.pop("messages")
                        identity = text_sha256(f"{POLICY}/{REVISION}/{title}/{i_p}/{i_c}/{character}/{i_w}")
                        source = {"dataset": downloaded["dataset"], "revision": REVISION, "file": name,
                                  "book": title, "plot_index": i_p, "conversation_index": i_c,
                                  "chapter": plot.get("chapter"), "chunk": plot.get("i_chunk"),
                                  "character": character, "window_index": i_w, **example,
                                  "scene_sha256": key, "book_selection_policy": BOOK_POLICY,
                                  "gutenberg_ids": [e["id"] for e in book["editions"]],
                                  "official_ood_book": is_test}
                        rows.append({"example_id": identity, "split_group": f"coser-book:{source_book_key(title)}",
                                     "split": "test" if is_test else "unassigned", "source": source, "messages": messages})
        book_report.append({"book": title, "is_test": is_test, "source_scenes": scenes, "examples": len(rows) - count})
        print(f"SFT {n}/{len(selected)}: {title}: {len(rows) - count} examples", flush=True)
    sft = [r for r in rows if r["split"] != "test"]
    test = [r for r in rows if r["split"] == "test"]
    validation_groups_set, _ = validation_groups(sft, config.seed, config.validation_fraction, config.split_group_field)
    for row in sft:
        row["split"] = "validation" if group_id(row, config.role, config.split_group_field) in validation_groups_set else "train"
    train, valid = [r for r in sft if r["split"] == "train"], [r for r in sft if r["split"] == "validation"]
    if min(len(train), len(valid), len(test)) < 2:
        raise ValueError("A dataset split is empty or too small")
    stats = {split: statistics(part, serializer) for split, part in [("train", train), ("validation", valid), ("test", test)]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".coser-sft-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "ready"
        staging.mkdir()
        for filename, part in [("all.jsonl", rows), ("sft.jsonl", sft), ("train.jsonl", train),
                               ("validation.jsonl", valid), ("test.jsonl", test)]:
            write_jsonl(staging / filename, part)
        atomic_json(staging / "book_catalog.json", catalog)
        export_prompts(valid, serializer, "validation", staging)
        export_prompts(test, serializer, "test", staging)
        atomic_json(staging / "splits.json", {split: sorted({r["split_group"] for r in part})
                                              for split, part in [("train", train), ("validation", valid), ("test", test)]})
        report = {"policy": POLICY, "book_policy": BOOK_POLICY, "revision": REVISION,
                  "download_manifest_sha256": sha256(root / "download_manifest.json"),
                  "seed": config.seed, "validation_fraction": config.validation_fraction,
                  "max_seq_length": 2048, "max_reply_including_eot": MAX_REPLY,
                  "statistics": stats, "books": book_report, "exclusions": dict(exclusions),
                  "source_scenes_before_views": len(seen), "total_examples": len(rows),
                  "optimizer_steps_one_epoch": math.ceil(len(train) / config.gradient_accumulation),
                  "one_epoch_assistant_target_tokens": stats["train"]["assistant_target_tokens"],
                  "training_started": False, "code_sha256": sha256(Path(__file__)),
                  "conversion_sha256": sha256(Path(__file__).with_name("coser_data.py")),
                  "tokenizer": {"id": config.tokenizer_id, "revision": config.tokenizer_revision},
                  "notes": ["All eligible records; no duplication/upsampling to meet a numeric target.",
                            "Official OOD books are test-only. Other official held-out plots are excluded.",
                            "One book belongs to one split; spoken-scene duplicates removed before character views.",
                            "Book-level PG selection does not establish exact CoSER translation/edition provenance.",
                            "This is dialogue SFT; it does not train a memory writer or prove memory-conditioned personality."],
                  "files": {p.name: sha256(p) for p in staging.iterdir()}}
        atomic_json(staging / "report.json", report)
        staging.rename(destination)
    print(json.dumps({"statistics": stats, "exclusions": dict(exclusions),
                      "optimizer_steps_one_epoch": report["optimizer_steps_one_epoch"]}, ensure_ascii=False, indent=2), flush=True)
    return report


def re_control(text):
    return any(marker in text for marker in ("[", "]", "<|", "<system", "<think", "_thinking>"))


def check(root=DEFAULT_ROOT, config_path=CONFIG):
    config = load_config(config_path)
    directory = config.path("dataset").parent
    root = Path(root)
    report = json.loads((directory / "report.json").read_text())
    if report["policy"] != POLICY or report["revision"] != REVISION:
        raise ValueError("SFT policy/revision changed")
    for name, digest in report["files"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Export checksum mismatch: {name}")
    rows = read_rows(directory / "all.jsonl")
    sft = read_rows(config.path("dataset"))
    metadata = {r["example_id"]: r for r in rows}
    assert len(metadata) == len(rows)
    groups = {split: {r["split_group"] for r in rows if r["split"] == split} for split in ("train", "validation", "test")}
    assert groups["train"].isdisjoint(groups["validation"] | groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])
    scenes = {split: {r["source"]["scene_sha256"] for r in rows if r["split"] == split} for split in groups}
    assert scenes["train"].isdisjoint(scenes["validation"] | scenes["test"])
    assert scenes["validation"].isdisjoint(scenes["test"])
    assert {r["example_id"] for r in sft} == {r["example_id"] for r in rows if r["split"] != "test"}
    downloaded = json.loads((root / "download_manifest.json").read_text())
    assert sha256(root / "download_manifest.json") == report["download_manifest_sha256"]
    tokenizer = load_tokenizer(config, offline=True)
    serializer = ChatSerializer(tokenizer)
    _, train_tensors, valid_tensors = load_prepared(config)
    for split, items in [("train", train_tensors), ("validation", valid_tensors)]:
        expected = {r["example_id"] for r in sft if r["split"] == split}
        assert expected == {sft[item["row"] - 1]["example_id"] for item in items}
        for item in items:
            encoded, labels = serializer.encode(sft[item["row"] - 1]["messages"])
            assert torch.equal(encoded, item["input_ids"]) and torch.equal(labels, item["labels"])
    catalog = json.loads((directory / "book_catalog.json").read_text())
    assert sha256(root.parent / "pg_catalog.csv") == catalog["pg_catalog_sha256"]
    books = {b["book"]: b for b in catalog["books"] if b["status"] == "selected"}
    for book in books.values():
        assert book["editions"] and len(book["author_evidence"]) == 1
        for edition in book["editions"]:
            assert edition["status"] == "public_domain_us"
            for suffix, field in [("html", "card_sha256"), ("txt", "text_sha256")]:
                assert sha256(root.parent / "gutenberg/editions" / f"{edition['id']}.{suffix}") == edition[field]
    raw_cache, scenes_cache = {}, {}
    for row in rows:
        source, messages = row["source"], row["messages"]
        assert not any(re_control(m["content"]) for m in messages)
        assert messages[0]["role"] == "system" and messages[-1]["role"] == "assistant"
        assert [m["role"] for m in messages[1:]] == ["user", "assistant"] * ((len(messages) - 1) // 2)
        assert len(serializer.encode(messages)[0]) <= 2048
        assert all(len(tokenizer.encode(m["content"], add_special_tokens=False)) + 1 <= MAX_REPLY for m in messages if m["role"] == "assistant")
        name = source["file"]
        if name not in raw_cache:
            assert sha256(root / "raw" / name) == downloaded["files"][name]["sha256"]
            raw_cache[name] = json.loads((root / "raw" / name).read_text())
        data = raw_cache[name]
        assert (row["split"] == "test") == source["official_ood_book"]
        assert row["split"] == "test" or source["plot_index"] < data["split_plot_index"]
        assert books[source["book"]]["status"] == "selected"
        plot = data["plots"][source["plot_index"]]
        conversation = plot["conversation"][source["conversation_index"]]
        scene_key = (name, source["plot_index"], source["conversation_index"])
        if scene_key not in scenes_cache:
            scenes_cache[scene_key] = clean_utterances(conversation)
        selected = [u for u in scenes_cache[scene_key] if source["utterance_start"] <= u["utterance_index"] <= source["utterance_end"]]
        recovered = [{"role": m["role"], "content": m["content"]} for pair in character_exchanges(selected, source["character"]) for m in pair]
        assert recovered == messages[1:]
        assert messages[0]["content"] == make_system(source["book"], source["character"], plot, conversation)
    for split in ("validation", "test"):
        cases = json.loads((directory / f"{split}_cases.json").read_text())["consciousness"]
        references = json.loads((directory / f"{split}_references.json").read_text())
        for case, reference in zip(cases, references, strict=True):
            row = metadata[reference["example_id"]]
            index = reference["assistant_message_index"]
            assert row["split"] == split and case["id"] == reference["id"]
            assert case["messages"] == row["messages"][:index]
            assert reference["reference"] == row["messages"][index]["content"]
            assert len(serializer.prompt(case["messages"])) + MAX_REPLY <= 2048
    result = {"status": "passed", "examples_checked": len(rows), "source_books_checked": len(raw_cache),
              "prepared_train": len(train_tensors), "prepared_validation": len(valid_tensors),
              "test_examples": report["statistics"]["test"]["examples"], "book_and_scene_overlap": 0,
              "training_started": False}
    atomic_json(directory / "check_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "check"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    (build if args.action == "build" else check)(args.root, args.config)
