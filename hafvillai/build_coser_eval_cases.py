"""Prepare validation prompts and separate references for the two CoSER adapters."""
import argparse
import json
from pathlib import Path

from training.config import ROOT, load_config
from training.data import read_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    templates = json.loads((ROOT/"hafvillai/examples/coser_checks.json").read_text())
    dialogue = json.loads((ROOT/"dataset-studio/data/coser/sft-v1/validation_cases.json").read_text())
    dialogue_refs = json.loads((ROOT/"dataset-studio/data/coser/sft-v1/validation_references.json").read_text())
    cases = {"consciousness": dialogue["consciousness"][:2], "subconsciousness": []}
    cases["consciousness"] += [c for c in templates["consciousness"]
                              if c["id"] in ("name_persona", "history_correction")]
    config = load_config(ROOT/"training/configs/coser_subconsciousness.json")
    rows = read_rows(config.path("dataset"))
    references = {"consciousness": dialogue_refs, "subconsciousness": []}
    seen = set()
    for row in rows:
        if row["split"] != "validation" or row["split_group"] in seen:
            continue
        seen.add(row["split_group"])
        name = f"coser_summary_validation_{len(seen):02d}"
        cases["subconsciousness"].append({"id": name, "format": "chat",
                                          "messages": row["messages"][:-1]})
        references["subconsciousness"].append({"case_id": name,"example_id": row["example_id"],
                                               "source": row["source"],
                                               "reference": row["messages"][-1]["content"]})
        if len(seen) == 2:
            break
    for case in templates["subconsciousness"]:
        if case["id"] not in ("summary_promise", "summary_plan_not_event"):
            continue
        copied = json.loads(json.dumps(case))
        copied["messages"][0] = rows[0]["messages"][0]
        cases["subconsciousness"].append(copied)
    if any(len(v) != 4 for v in cases.values()):
        raise ValueError("Expected four prompts for each role")
    for name, data in (("cases.json",cases),("references.json",references)):
        (args.output_dir/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
    print("EVALUATION_CASES_READY",{k:len(v) for k,v in cases.items()},flush=True)


if __name__ == "__main__":
    main()
