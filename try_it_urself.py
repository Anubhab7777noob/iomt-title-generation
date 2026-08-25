"""
Quick manual test: paste any abstract/paragraph below and get a generated title.
Uses the same local Mistral setup as baseline_branch.py, just without needing
a test.json file -- for quick one-off tries.

Usage:
    python try_it_yourself.py
"""

import ollama

PROMPT_TEMPLATE = """You are given the abstract of a research paper. Generate a concise, \
academic-style title for this paper based solely on the abstract. \
Respond with ONLY the title text, no quotation marks, no explanation, no preamble.

Abstract:
{abstract}

Title:"""


def generate_title(abstract: str) -> str:
    response = ollama.generate(
        model="mistral",
        prompt=PROMPT_TEMPLATE.format(abstract=abstract),
        options={"temperature": 0.3},
    )
    return response["response"].strip().strip('"')


if __name__ == "__main__":
    # <-- Replace this with any paragraph/abstract you want to try
    my_abstract = """
    This paper presents a low-power wireless sensor network designed for continuous glucose monitoring in diabetic patients. The system integrates a biosensor with a microcontroller and Bluetooth Low Energy module to transmit real-time glucose readings to a companion smartphone application. Machine learning algorithms are employed to predict hypoglycemic events up to 30 minutes in advance, enabling timely intervention. Clinical trials involving 45 participants demonstrated a prediction accuracy of 92% with minimal false alarms, suggesting strong potential for improving diabetes management through proactive alerting.
    """

    title = generate_title(my_abstract)
    print("Generated title:")
    print(title)