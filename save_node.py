"""
Save node.
Writes the final merged paper list to both JSON and CSV.
"""

import json
import csv
from schema import PaperMetadata


def save_papers(papers: list[PaperMetadata], json_path: str = "papers.json", csv_path: str = "papers.csv"):
    # JSON: full fidelity, including nested authors list
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in papers], f, indent=2, ensure_ascii=False)

    # CSV: flatten authors list into a semicolon-joined string
    if papers:
        fieldnames = list(papers[0].model_dump().keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for p in papers:
                row = p.model_dump()
                row["authors"] = "; ".join(row["authors"])
                writer.writerow(row)

    print(f"Saved {len(papers)} papers to {json_path} and {csv_path}")
