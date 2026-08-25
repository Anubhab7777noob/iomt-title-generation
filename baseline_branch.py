"""
Stage 2, Branch 1: Baseline.

Sends each test-set abstract to local Mistral (via Ollama) with a plain prompt asking
for a title. No fine-tuning, no retrieval-augmented context -- this is the control
group that the fine-tune and RAG branches will be compared against.

Each call is independent: Ollama's Python client (unlike the interactive `ollama run`
CLI) does NOT carry conversation memory between calls unless you explicitly pass
previous messages back in. Every abstract is scored fresh, with no leakage from
prior predictions.

Requires: `ollama` running locally with the `mistral` model pulled
    (ollama pull mistral)

Usage:
    python baseline_branch.py
"""

import json
import ollama

TEST_PATH = "test.json"
OUTPUT_PATH = "baseline_predictions.json"
MODEL_NAME = "mistral"

PROMPT_TEMPLATE = """You are given the abstract of a research paper. Generate a concise, \
academic-style title for this paper based solely on the abstract. \
Respond with ONLY the title text, no quotation marks, no explanation, no preamble.

Abstract:
{This paper presents a low-power wireless sensor network designed for continuous glucose monitoring in diabetic patients. The system integrates a biosensor with a microcontroller and Bluetooth Low Energy module to transmit real-time glucose readings to a companion smartphone application. Machine learning algorithms are employed to predict hypoglycemic events up to 30 minutes in advance, enabling timely intervention. Clinical trials involving 45 participants demonstrated a prediction accuracy of 92% with minimal false alarms, suggesting strong potential for improving diabetes management through proactive alerting.}

Title:"""


def generate_title(abstract: str) -> str:
    """
    Sends a single abstract to Mistral via Ollama and returns the generated title.
    This is a fresh, independent call each time -- no memory of prior calls.
    """
    response = ollama.generate(
        model=MODEL_NAME,
        prompt=PROMPT_TEMPLATE.format(abstract=abstract),
        options={
            "temperature": 0.3,  # low temperature: we want consistent, focused titles, not creative variation
        },
    )
    return response["response"].strip().strip('"')  # strip stray quotes the model sometimes adds


def run_baseline():
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
    run_baseline()
