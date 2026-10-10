"""
preprocess.py

Converts the raw data/proverbs.jsonl file (instruction/input/output triples)
into a single formatted prompt string per example, ready for causal-LM
fine-tuning (e.g. an Alpaca-style prompt template).

Usage:
    python scripts/preprocess.py \
        --input data/proverbs.jsonl \
        --output data/proverbs_formatted.jsonl
"""

import argparse
import json
from pathlib import Path

PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)


def format_example(example: dict) -> str:
    if example.get("input"):
        return PROMPT_WITH_INPUT.format(**example)
    return PROMPT_NO_INPUT.format(**example)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/proverbs.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/proverbs_formatted.jsonl")
    )
    args = parser.parse_args()

    examples = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))

    with args.output.open("w", encoding="utf-8") as f:
        for ex in examples:
            text = format_example(ex)
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} formatted examples to {args.output}")


if __name__ == "__main__":
    main()
