"""
arXiv agent node.
Uses the official `arxiv` python package (wraps arXiv's public Atom API). No API key needed.
Docs: https://info.arxiv.org/help/api/index.html

NOTE: arXiv rate-limits aggressively. This module adds a small delay before each request
and retries with backoff on HTTP 429 (Too Many Requests), since without it, rapid back-to-back
queries (like when looping over multiple search terms) will get blocked.
"""

import time
import arxiv
from schema import PaperMetadata

# Reusable client with built-in delay between internal pagination requests.
# delay_seconds is arxiv package's own throttle between paged requests to the SAME query.
_client = arxiv.Client(delay_seconds=3.0, num_retries=3)


def fetch_arxiv_papers(query: str, max_results: int = 20, max_attempts: int = 4) -> list[PaperMetadata]:
    """
    Query arXiv and return a list of PaperMetadata.
    `query` example: "IoT healthcare monitoring"
    Retries with exponential backoff if arXiv responds with 429 (rate limited).
    """
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    attempt = 0
    while True:
        try:
            papers = []
            for result in _client.results(search):
                papers.append(
                    PaperMetadata(
                        title=result.title.strip().replace("\n", " "),
                        authors=[a.name for a in result.authors],
                        journal=result.journal_ref,  # often None on arXiv, that's expected
                        abstract=result.summary.strip().replace("\n", " ") if result.summary else None,
                        publish_date=result.published.strftime("%Y-%m-%d") if result.published else None,
                        doi=result.doi,
                        url=result.entry_id,
                        source="arxiv",
                    )
                )
            return papers
        except arxiv.HTTPError as e:
            attempt += 1
            if attempt >= max_attempts:
                print(f"[arxiv] Giving up on query '{query}' after {attempt} attempts: {e}")
                return []  # fail gracefully -- return empty instead of crashing the whole pipeline
            wait = 10 * attempt  # 10s, 20s, 30s...
            print(f"[arxiv] Rate limited on '{query}', retrying in {wait}s (attempt {attempt}/{max_attempts})...")
            time.sleep(wait)


# Quick standalone test
if __name__ == "__main__":
    results = fetch_arxiv_papers("IoT healthcare monitoring", max_results=5)
    for p in results:
        print(f"- {p.title} ({p.publish_date}) [{p.source}]")
    print(f"\nTotal: {len(results)} papers")
