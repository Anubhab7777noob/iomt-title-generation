"""
Stage 2, Branch 2: Fine-tuned.

Sends each test-set abstract to the LOCAL fine-tuned Mistral model (trained via
Unsloth on Colab, converted to GGUF, loaded into Ollama as "mistral-finetuned").

Same prompt template and structure as baseline_branch.py, so the only variable
that changes between the two branches is which model answers -- keeping the
comparison fair.

Requires: the "mistral-finetuned" model already created in Ollama
    (ollama create mistral-finetuned -f Modelfile)

Usage:
    python finetune_branch.py
"""

import json
import ollama

TEST_PATH = "test.json"
OUTPUT_PATH = "finetune_predictions.json"
MODEL_NAME = "mistral-finetuned"

PROMPT_TEMPLATE = """You are given the abstract of a research paper. Generate a concise, \
academic-style title for this paper based solely on the abstract. \
Respond with ONLY the title text, no quotation marks, no explanation, no preamble.

Abstract:
{abstract}

Title:"""


def generate_title(abstract: str) -> str:
    """
    Sends a single abstract to the fine-tuned model via Ollama and returns the
    generated title. Each call is independent -- no memory of prior calls.
    """
    response = ollama.generate(
        model=MODEL_NAME,
        prompt=PROMPT_TEMPLATE.format(abstract=abstract),
        options={
            "temperature": 0.3,
        },
    )
    return response["response"].strip().strip('"')


def run_finetune_branch():
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_papers = json.load(f)

    print(f"Loaded {len(test_papers)} test papers from {TEST_PATH}")
    results = []

    for i, paper in enumerate(test_papers, start=1):
        real_title = paper["title"]
        abstract = paper["abstract"]

        predicted_title = generate_title(abstract)

        results.append({
            "serial_no": paper.get("serial_no"),
            "real_title": real_title,
            "predicted_title": predicted_title,
        })

        print(f"[{i}/{len(test_papers)}] Real: {real_title[:60]}...")
        print(f"           Predicted: {predicted_title[:60]}...")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_finetune_branch()
