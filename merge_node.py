"""
Merge node.
Combines paper lists from all agent nodes, dedupes them, and assigns serial numbers.

Dedupe strategy:
1. Primary key: DOI (case-insensitive, exact match) -- most reliable when present.
2. Fallback: normalized title match (lowercase, strip punctuation/whitespace) for papers
   with no DOI, since arXiv preprints often lack one.
"""

import re
from schema import PaperMetadata


def _normalize_title(title: str) -> str:
    title = title.lower().strip()
    title = re.sub(r"[^\w\s]", "", title)  # strip punctuation
    title = re.sub(r"\s+", " ", title)     # collapse whitespace
    return title


def merge_and_dedupe(*paper_lists: list[PaperMetadata]) -> list[PaperMetadata]:
    """
    Takes any number of paper lists (one per agent) and returns a single
    deduped, serial-numbered list.
    """
    seen_dois = set()
    seen_titles = set()
    merged: list[PaperMetadata] = []

    for paper_list in paper_lists:
        for paper in paper_list:
            doi_key = paper.doi.strip().lower() if paper.doi else None
            title_key = _normalize_title(paper.title)

            if doi_key and doi_key in seen_dois:
                continue
            if not doi_key and title_key in seen_titles:
                continue

            if doi_key:
                seen_dois.add(doi_key)
            seen_titles.add(title_key)

            merged.append(paper)

    # Assign serial numbers after dedupe
    for i, paper in enumerate(merged, start=1):
        paper.serial_no = i

    return merged
