"""
PubMed agent node.
Uses NCBI E-utilities (esearch -> efetch), no API key required (rate limit: ~3 req/sec without a key).
Docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
"""

import requests
import xml.etree.ElementTree as ET
import time
from schema import PaperMetadata

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def fetch_pubmed_papers(query: str, max_results: int = 20) -> list[PaperMetadata]:
    """
    Query PubMed and return a list of PaperMetadata.
    Two-step: esearch gets PMIDs matching the query, efetch pulls full records for those PMIDs.
    """
    # Step 1: get matching PMIDs
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
    }
    search_resp = requests.get(ESEARCH_URL, params=search_params, timeout=15)
    search_resp.raise_for_status()
    pmids = search_resp.json().get("esearchresult", {}).get("idlist", [])

    if not pmids:
        return []

    time.sleep(0.4)  # stay under the ~3 req/sec keyless rate limit

    # Step 2: fetch full records for those PMIDs
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    fetch_resp = requests.get(EFETCH_URL, params=fetch_params, timeout=15)
    fetch_resp.raise_for_status()

    root = ET.fromstring(fetch_resp.content)
    papers = []

    for article in root.findall(".//PubmedArticle"):
        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else None
        if not title:
            continue

        abstract_parts = article.findall(".//Abstract/AbstractText")
        abstract = " ".join("".join(a.itertext()) for a in abstract_parts).strip() or None

        journal_el = article.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else None

        authors = []
        for author in article.findall(".//AuthorList/Author"):
            last = author.find("LastName")
            fore = author.find("ForeName")
            if last is not None and fore is not None:
                authors.append(f"{fore.text} {last.text}")
            elif last is not None:
                authors.append(last.text)

        year_el = article.find(".//PubDate/Year")
        publish_date = year_el.text if year_el is not None else None

        doi = None
        for id_el in article.findall(".//ArticleIdList/ArticleId"):
            if id_el.get("IdType") == "doi":
                doi = id_el.text

        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else None
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None

        papers.append(
            PaperMetadata(
                title=title,
                authors=authors,
                journal=journal,
                abstract=abstract,
                publish_date=publish_date,
                doi=doi,
                url=url,
                source="pubmed",
            )
        )

    return papers


# Quick standalone test
if __name__ == "__main__":
    results = fetch_pubmed_papers("IoT healthcare monitoring", max_results=5)
    for p in results:
        print(f"- {p.title} ({p.publish_date}) [{p.source}]")
    print(f"\nTotal: {len(results)} papers")
