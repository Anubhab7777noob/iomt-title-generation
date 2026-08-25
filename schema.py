"""
Shared Pydantic schema for paper metadata.
Every agent node (arXiv, PubMed, CrossRef) must return a list of PaperMetadata objects
using this exact schema, so the merge node can combine them without extra transformation.
"""

from pydantic import BaseModel, Field
from typing import Optional


class PaperMetadata(BaseModel):
    serial_no: Optional[int] = Field(
        default=None,
        description="Assigned later by the merge node, not by individual agents."
    )
    title: str = Field(description="Full title of the paper.")
    authors: list[str] = Field(default_factory=list, description="List of author names.")
    journal: Optional[str] = Field(default=None, description="Journal or venue name.")
    abstract: Optional[str] = Field(default=None, description="Abstract text, if available.")
    publish_date: Optional[str] = Field(default=None, description="Publication date as string (YYYY-MM-DD or YYYY).")
    doi: Optional[str] = Field(default=None, description="DOI, used as the primary dedupe key.")
    url: Optional[str] = Field(default=None, description="Link to the paper.")
    source: str = Field(description="Which agent found this paper: 'arxiv', 'pubmed', or 'crossref'.")


class PaperCollection(BaseModel):
    """Wrapper used to pass a batch of papers between graph nodes."""
    papers: list[PaperMetadata] = Field(default_factory=list)
