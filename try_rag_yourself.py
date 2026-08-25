"""
Quick manual test for the RAG branch: paste any abstract and see which train
examples get retrieved, plus the generated title -- without running the full
25-abstract test set.

Requires: train.json in the same folder, and "nomic-embed-text" + "mistral"
pulled in Ollama.

Usage:
    python try_rag_yourself.py
"""

import json
import math
import ollama

TRAIN_PATH = "train.json"
GENERATION_MODEL = "mistral"
EMBEDDING_MODEL = "nomic-embed-text"
TOP_K = 3

PROMPT_TEMPLATE = """You are given the abstract of a research paper, along with a few \
examples of similar papers and their real titles. Use the examples as a style/content \
guide, then generate a concise, academic-style title for the NEW abstract at the bottom. \
Respond with ONLY the title text, no quotation marks, no explanation, no preamble.

{examples}

Now generate a title for this NEW abstract:
{abstract}

Title:"""

EXAMPLE_BLOCK_TEMPLATE = """Example {n}:
Abstract: {abstract}
Title: {title}
"""


def embed_text(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


if __name__ == "__main__":
    # <-- Replace this with any abstract you want to try
    my_abstract = """This paper presents a low-power wireless sensor network designed for continuous glucose monitoring in diabetic patients. The system integrates a biosensor with a microcontroller and Bluetooth Low Energy module to transmit real-time glucose readings to a companion smartphone application. Machine learning algorithms are employed to predict hypoglycemic events up to 30 minutes in advance, enabling timely intervention. Clinical trials involving 45 participants demonstrated a prediction accuracy of 92% with minimal false alarms, suggesting strong potential for improving diabetes management through proactive alerting.
    """

    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        train_papers = json.load(f)

    print(f"Embedding {len(train_papers)} train abstracts (one-time cost)...")
    train_index = []
    for paper in train_papers:
        embedding = embed_text(paper["abstract"])
        train_index.append({
            "title": paper["title"],
            "abstract": paper["abstract"],
            "embedding": embedding,
        })

    query_embedding = embed_text(my_abstract)
    scored = [
        (cosine_similarity(query_embedding, entry["embedding"]), entry)
        for entry in train_index
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    top_examples = [entry for _, entry in scored[:TOP_K]]

    print("\nTop retrieved examples:")
    for score, entry in scored[:TOP_K]:
        print(f"  ({score:.3f}) {entry['title']}")

    example_blocks = "\n".join(
        EXAMPLE_BLOCK_TEMPLATE.format(n=i, abstract=ex["abstract"], title=ex["title"])
        for i, ex in enumerate(top_examples, start=1)
    )
    prompt = PROMPT_TEMPLATE.format(examples=example_blocks, abstract=my_abstract)

    response = ollama.generate(
        model=GENERATION_MODEL,
        prompt=prompt,
        options={"temperature": 0.3},
    )
    title = response["response"].strip().strip('"')

    print("\nGenerated title:")
    print(title)
