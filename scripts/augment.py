"""
augment.py

Expands the curated seed dataset (data/proverbs.jsonl) by generating
paraphrased instruction variants for each proverb, while keeping the
output (the proverb, translation, and meaning) unchanged.

This is a standard instruction-tuning data augmentation technique: it
does NOT invent new proverbs, it teaches the model to respond correctly
to differently-worded requests for the same proverb. The language and
topic are extracted from the original instruction with a regex, then
re-inserted into a set of alternate templates.

Usage:
    python scripts/augment.py \
        --input data/proverbs.jsonl \
        --output data/proverbs_augmented.jsonl \
        --variants_per_example 1
"""

import argparse
import json
import random
import re
from pathlib import Path

LANGUAGES = ["Yoruba", "Igbo", "Hausa"]

TOPIC_PATTERN = re.compile(
    r"(?:proverb|saying)\s+(?:about|related to)\s+(.+?)(?:\s+and explain.*)?[\.\?]?$",
    re.IGNORECASE,
)

TEMPLATES = [
    "What's {article} {lang} proverb about {topic}?",
    "Tell me {article} {lang} saying about {topic} and what it means.",
    "I'd like {article} {lang} proverb related to {topic}, with an explanation.",
    "Do you know {article} {lang} proverb on the subject of {topic}?",
    "Teach me {article} {lang} proverb about {topic}.",
]

VOWEL_LEAD_LANGUAGES = {"Igbo"}


def article_for(lang: str) -> str:
    return "an" if lang in VOWEL_LEAD_LANGUAGES else "a"


def extract_language(instruction: str):
    for lang in LANGUAGES:
        if lang.lower() in instruction.lower():
            return lang
    return None


def extract_topic(instruction: str):
    match = TOPIC_PATTERN.search(instruction)
    if match:
        return match.group(1).strip().rstrip(".")
    return None


def make_variant(instruction: str, rng: random.Random) -> str | None:
    lang = extract_language(instruction)
    topic = extract_topic(instruction)
    if not lang or not topic:
        return None
    template = rng.choice(TEMPLATES)
    return template.format(lang=lang, topic=topic, article=article_for(lang))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/proverbs.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/proverbs_augmented.jsonl")
    )
    parser.add_argument(
        "--variants_per_example",
        type=int,
        default=1,
        help="How many paraphrased instruction variants to add per original example.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    originals = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                originals.append(json.loads(line))

    augmented = list(originals)
    skipped = 0

    for ex in originals:
        used_templates = set()
        for _ in range(args.variants_per_example):
            variant_instruction = make_variant(ex["instruction"], rng)
            if variant_instruction is None or variant_instruction in used_templates:
                skipped += 1
                continue
            used_templates.add(variant_instruction)
            augmented.append(
                {
                    "instruction": variant_instruction,
                    "input": ex.get("input", ""),
                    "output": ex["output"],
                }
            )

    with args.output.open("w", encoding="utf-8") as f:
        for ex in augmented:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(
        f"{len(originals)} original examples -> {len(augmented)} total after "
        f"augmentation ({skipped} skipped, no language/topic match). "
        f"Wrote {args.output}"
    )


if __name__ == "__main__":
    main()
