"""
Stage 2, Branch 3: RAG (Retrieval-Augmented Generation).

For each test abstract:
  1. Embed it using Ollama's nomic-embed-text model.
  2. Find the 3 most similar abstracts from the TRAIN set (cosine similarity).
  3. Build a prompt that includes those 3 similar (abstract, title) pairs as examples.
  4. Ask Mistral to generate a title for the test abstract, using those examples as context.

No model weights are changed here -- this is "in-context learning" via retrieval,
not fine-tuning. The train set acts as a knowledge base to pull relevant examples from.

Requires: `ollama` running locally with `mistral` and `nomic-embed-text` pulled
    (ollama pull mistral && ollama pull nomic-embed-text)

Usage:
    python rag_branch.py
"""

import json
import math
import ollama

TRAIN_PATH = "train.json"
TEST_PATH = "test.json"
OUTPUT_PATH = "rag_predictions.json"

GENERATION_MODEL = "mistral"
EMBEDDING_MODEL = "nomic-embed-text"
TOP_K = 3  # number of similar train examples to retrieve per test abstract

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
    """Returns the embedding vector for a piece of text using Ollama's embedding model."""
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_train_index(train_papers: list[dict]) -> list[dict]:
    """
    Embeds every train abstract once up front, so we don't re-embed the whole
    train set for every single test query (that would be 100x slower).
    """
    print(f"Embedding {len(train_papers)} train abstracts (one-time cost)...")
    indexed = []
    for i, paper in enumerate(train_papers, start=1):
        embedding = embed_text(paper["abstract"])
        indexed.append({
            "title": paper["title"],
            "abstract": paper["abstract"],
            "embedding": embedding,
        })
        if i % 10 == 0 or i == len(train_papers):
            print(f"  Embedded {i}/{len(train_papers)}")
    return indexed


def retrieve_top_k(query_embedding: list[float], train_index: list[dict], k: int) -> list[dict]:
    scored = [
        (cosine_similarity(query_embedding, entry["embedding"]), entry)
        for entry in train_index
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in scored[:k]]


def generate_title_with_context(abstract: str, examples: list[dict]) -> str:
    example_blocks = "\n".join(
        EXAMPLE_BLOCK_TEMPLATE.format(n=i, abstract=ex["abstract"], title=ex["title"])
        for i, ex in enumerate(examples, start=1)
    )
    prompt = PROMPT_TEMPLATE.format(examples=example_blocks, abstract=abstract)

    response = ollama.generate(
        model=GENERATION_MODEL,
        prompt=prompt,
        options={"temperature": 0.3},
    )
    return response["response"].strip().strip('"')


def run_rag():
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        train_papers = json.load(f)
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_papers = json.load(f)

    print(f"Loaded {len(train_papers)} train papers, {len(test_papers)} test papers")

    train_index = build_train_index(train_papers)

    results = []
    for i, paper in enumerate(test_papers, start=1):
        real_title = paper["title"]
        abstract = paper["abstract"]

        query_embedding = embed_text(abstract)
        top_examples = retrieve_top_k(query_embedding, train_index, TOP_K)
        predicted_title = generate_title_with_context(abstract, top_examples)

        results.append({
            "serial_no": paper.get("serial_no"),
            "real_title": real_title,
            "predicted_title": predicted_title,
            "retrieved_examples": [ex["title"] for ex in top_examples],  # for debugging/inspection
        })

        print(f"[{i}/{len(test_papers)}] Real: {real_title[:60]}...")
        print(f"           Predicted: {predicted_title[:60]}...")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_rag()
