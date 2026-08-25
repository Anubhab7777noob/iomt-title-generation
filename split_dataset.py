"""
Splits papers.json into train.json (80%) and test.json (20%).

This split is the single source of truth for Stage 2 -- every branch (baseline,
fine-tune, RAG) must load train.json / test.json from here rather than re-splitting
on their own, so all three are evaluated on the identical held-out set.

Usage:
    python split_dataset.py
"""

import json
import random

SEED = 42          # fixed seed so the split is reproducible if you re-run this
TRAIN_RATIO = 0.8

INPUT_PATH = "papers.json"
TRAIN_OUTPUT_PATH = "train.json"
TEST_OUTPUT_PATH = "test.json"


def split_dataset():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        papers = json.load(f)

    print(f"Loaded {len(papers)} papers from {INPUT_PATH}")

    # Shuffle with a fixed seed so this split is reproducible across runs,
    # rather than getting a different random split every time you execute this.
    rng = random.Random(SEED)
    shuffled = papers.copy()
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * TRAIN_RATIO)
    train_papers = shuffled[:split_idx]
    test_papers = shuffled[split_idx:]

    with open(TRAIN_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(train_papers, f, indent=2, ensure_ascii=False)

    with open(TEST_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(test_papers, f, indent=2, ensure_ascii=False)

    print(f"Train set: {len(train_papers)} papers -> {TRAIN_OUTPUT_PATH}")
    print(f"Test set:  {len(test_papers)} papers -> {TEST_OUTPUT_PATH}")


if __name__ == "__main__":
    split_dataset()
