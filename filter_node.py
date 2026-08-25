"""
Filter node.
Drops any paper with a missing or empty abstract, since Stage 2 (abstract -> title
generation) requires a real abstract for every row. Common with CrossRef results,
which frequently omit abstracts.
"""

from schema import PaperMetadata


def filter_missing_abstracts(papers: list[PaperMetadata]) -> list[PaperMetadata]:
    """
    Returns only papers with a non-empty abstract.
    Re-numbers serial_no after filtering so IDs stay contiguous (1, 2, 3, ...).
    """
    filtered = [p for p in papers if p.abstract and p.abstract.strip()]

    for i, paper in enumerate(filtered, start=1):
        paper.serial_no = i

    dropped = len(papers) - len(filtered)
    if dropped:
        print(f"[filter] Dropped {dropped} papers with missing/empty abstracts "
              f"({len(filtered)} remain).")

    return filtered
