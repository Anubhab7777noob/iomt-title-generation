"""
CrossRef agent node.
Uses CrossRef's public REST API, no API key required.
Docs: https://api.crossref.org/swagger-ui/index.html
"""

import requests
from schema import PaperMetadata

CROSSREF_URL = "https://api.crossref.org/works"


def fetch_crossref_papers(query: str, max_results: int = 20) -> list[PaperMetadata]:
    """
    Query CrossRef and return a list of PaperMetadata.
    """
    params = {
        "query": query,
        "rows": max_results,
        # Being a polite API citizen: identify yourself so CrossRef doesn't throttle you.
        # Swap in a real contact email before running at scale.
        "mailto": "your_email@example.com",
    }
    resp = requests.get(CROSSREF_URL, params=params, timeout=15)
    resp.raise_for_status()
    items = resp.json().get("message", {}).get("items", [])

    papers = []
    for item in items:
        title_list = item.get("title", [])
        title = title_list[0].strip() if title_list else None
        if not title:
            continue

        authors = []
        for a in item.get("author", []):
            given = a.get("given", "")
            family = a.get("family", "")
            name = f"{given} {family}".strip()
            if name:
                authors.append(name)

        container = item.get("container-title", [])
        journal = container[0] if container else None

        # CrossRef rarely includes abstracts; when present they're JATS-tagged XML-ish text
        abstract = item.get("abstract")
        if abstract:
            # strip simple XML tags like <jats:p>
            import re
            abstract = re.sub(r"<[^>]+>", "", abstract).strip()

        date_parts = item.get("published", {}).get("date-parts", [[None]])
        publish_date = "-".join(str(p) for p in date_parts[0] if p) if date_parts[0][0] else None

        papers.append(
            PaperMetadata(
                title=title,
                authors=authors,
                journal=journal,
                abstract=abstract,
                publish_date=publish_date,
                doi=item.get("DOI"),
                url=item.get("URL"),
                source="crossref",
            )
        )

    return papers


# Quick standalone test
if __name__ == "__main__":
    results = fetch_crossref_papers("IoT healthcare monitoring", max_results=5)
    for p in results:
        print(f"- {p.title} ({p.publish_date}) [{p.source}]")
    print(f"\nTotal: {len(results)} papers")
