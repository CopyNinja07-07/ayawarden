"""Stage 1 (CuratorKIT): quality-gate the raw rows and export trainer-ready data with provenance.

The prompt is flattened into one user turn (policy + task + tool output), so the base model and the
fine-tuned model see exactly the same input at evaluation time.
Outputs: data/curated/{train,val}/  (sft_messages.jsonl, sft_sharegpt.jsonl, manifest.json, ...)
"""
from __future__ import annotations

import json
from pathlib import Path

from curatorkit import Curator, CuratorConfig

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
STAGE = ROOT / "data" / "staging"
CUR = ROOT / "data" / "curated"


def flatten(row: dict) -> dict:
    system, user = row["messages"][0]["content"], row["messages"][1]["content"]
    task, tool = user.split("\n\n<tool_output>", 1)
    return {
        "id": row["id"],
        "instruction": f"{system}\n\n{task}",
        "input": "<tool_output>" + tool,
        "output": row["response"],
        "source_dataset": f"{row['family']}|{row['lang']}|{row['kind']}",
    }


def main() -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        rows = [json.loads(l) for l in open(RAW / f"{split}.jsonl")]
        stage_file = STAGE / f"{split}.jsonl"
        with open(stage_file, "w") as f:
            for r in rows:
                f.write(json.dumps(flatten(r), ensure_ascii=False) + "\n")
        result = Curator(CuratorConfig(
            name=f"ayawarden_{split}",
            dataset=str(stage_file),
            format="alpaca",
            field_mapping={"instruction": "instruction", "input": "input", "output": "output"},
            dedup="exact",          # templated data: exact dup removal only (near-dup would drop valid variants)
            clean=True,
            clean_transforms={"strip_html": False},  # keep tag-shaped injections + <tool_output> tags
            min_tokens=20,
            max_tokens=2048,
            export_formats=["messages", "sharegpt", "alpaca"],
            output_dir=str(CUR / split),
        )).run()
        result.print_summary()


if __name__ == "__main__":
    main()
