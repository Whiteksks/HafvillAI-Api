"""Prepare one clean, complete conversation per accepted CoSER scene, offline."""

import json
from pathlib import Path

from coser_data import clean_utterances
from data_utils import atomic_json, sha256

POLICY = "coser-complete-thoughtless-scenes-v1"
SOURCE_METADATA = ("example_id", "split", "split_group", "source")


def prepare_source(root):
    root = Path(root)
    accepted = root / "sft-v1/sft.jsonl"
    manifest = json.loads((root / "download_manifest.json").read_text())
    scenes, books, book_splits = {}, {}, {}
    for line in accepted.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["split"] == "test":
            continue
        if row["split"] not in {"train", "validation"}:
            raise ValueError("CoSER summary source requires train/validation splits")
        source = row["source"]
        if source["revision"] != manifest["revision"]:
            raise ValueError("CoSER source revision mismatch")
        previous = book_splits.setdefault(row["split_group"], row["split"])
        if previous != row["split"]:
            raise ValueError("A CoSER book appears in multiple splits")
        key = (source["file"], source["plot_index"], source["conversation_index"])
        scenes.setdefault(key, row)
    rows = []
    for (file, plot_index, conversation_index), row in sorted(scenes.items()):
        if file not in books:
            path = (root / "raw" / file).resolve()
            if not path.is_relative_to((root / "raw").resolve()):
                raise ValueError("Invalid CoSER source path")
            if sha256(path) != manifest["files"][file]["sha256"]:
                raise ValueError(f"Changed CoSER source: {file}")
            books[file] = json.loads(path.read_text(encoding="utf-8"))
        conversation = books[file]["plots"][plot_index]["conversation"][conversation_index]
        utterances = clean_utterances(conversation)
        text = "\n".join(f"{u['character']}: {u['content']}" for u in utterances)
        source = {k: row["source"][k] for k in
                  ("dataset", "revision", "file", "book", "plot_index", "conversation_index", "scene_sha256")}
        source.update(summary_policy=POLICY, raw_sha256=manifest["files"][file]["sha256"])
        rows.append({"example_id": f"coser-summary:{source['revision']}:{source['scene_sha256']}",
                     "split": row["split"], "split_group": row["split_group"], "source": source,
                     "messages": [{"role": "user", "content": text}]})
    if not rows:
        raise ValueError("No accepted CoSER conversations")
    directory = root / "summary-v1"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "dialogues.jsonl"
    temporary = output.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    temporary.replace(output)
    report = {"policy": POLICY, "accepted_sft_sha256": sha256(accepted),
              "download_manifest_sha256": sha256(root / "download_manifest.json"),
              "source_sha256": sha256(output), "records": len(rows),
              "splits": {split: sum(r["split"] == split for r in rows) for split in ("train", "validation")},
              "batch_size": 100, "batch_count": (len(rows) + 99) // 100,
              "test_included": False}
    atomic_json(directory / "report.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(prepare_source(Path(__file__).parent / "data/coser"), ensure_ascii=False, indent=2))
