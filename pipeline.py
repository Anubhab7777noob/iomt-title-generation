"""
Stage 1 pipeline: multi-agent paper search using LangGraph.

Graph shape:

    START
      |--> arxiv_node    --|
      |--> pubmed_node    --|--> merge_node --> save_node --> END
      |--> crossref_node  --|

Each agent node runs independently (LangGraph fans them out from START) and writes
its results into a shared list in the graph state. LangGraph merges parallel branch
outputs into state automatically via the reducer on `all_papers` (operator.add),
so no manual synchronization code is needed between the fan-out and the merge node.

Supports MULTIPLE search queries per source (e.g. "IoT healthcare monitoring",
"IoMT wearable sensors", "remote patient monitoring") for broader dataset coverage.
Each agent loops over all queries and collects results from each before returning.
"""

import operator
import time
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END

from schema import PaperMetadata
from agent_arxiv import fetch_arxiv_papers
from agent_pubmed import fetch_pubmed_papers
from agent_crossref import fetch_crossref_papers
from merge_node import merge_and_dedupe
from filter_node import filter_missing_abstracts
from save_node import save_papers


# ---- Graph state ----
class PipelineState(TypedDict):
    queries: list[str]              # multiple search queries for coverage
    max_results_per_source: int     # applied PER query, per source
    # operator.add lets multiple parallel nodes each append their own list;
    # LangGraph concatenates them automatically instead of overwriting.
    all_papers: Annotated[list[PaperMetadata], operator.add]
    merged_papers: list[PaperMetadata]
    filtered_papers: list[PaperMetadata]


# ---- Node functions ----
def arxiv_node(state: PipelineState) -> dict:
    papers = []
    for i, query in enumerate(state["queries"]):
        if i > 0:
            time.sleep(5)  # pause between distinct queries so arXiv doesn't rate-limit us
        papers.extend(fetch_arxiv_papers(query, state["max_results_per_source"]))
    return {"all_papers": papers}


def pubmed_node(state: PipelineState) -> dict:
    papers = []
    for query in state["queries"]:
        papers.extend(fetch_pubmed_papers(query, state["max_results_per_source"]))
    return {"all_papers": papers}


def crossref_node(state: PipelineState) -> dict:
    papers = []
    for query in state["queries"]:
        papers.extend(fetch_crossref_papers(query, state["max_results_per_source"]))
    return {"all_papers": papers}


def merge_node_fn(state: PipelineState) -> dict:
    merged = merge_and_dedupe(state["all_papers"])
    return {"merged_papers": merged}


def filter_node_fn(state: PipelineState) -> dict:
    filtered = filter_missing_abstracts(state["merged_papers"])
    return {"filtered_papers": filtered}


def save_node_fn(state: PipelineState) -> dict:
    save_papers(state["filtered_papers"])
    return {}


# ---- Build the graph ----
def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("arxiv_node", arxiv_node)
    graph.add_node("pubmed_node", pubmed_node)
    graph.add_node("crossref_node", crossref_node)
    graph.add_node("merge_node", merge_node_fn)
    graph.add_node("filter_node", filter_node_fn)
    graph.add_node("save_node", save_node_fn)

    # Fan out from START to all three agents in parallel
    graph.add_edge(START, "arxiv_node")
    graph.add_edge(START, "pubmed_node")
    graph.add_edge(START, "crossref_node")

    # All three feed into merge, then filter, then save
    graph.add_edge("arxiv_node", "merge_node")
    graph.add_edge("pubmed_node", "merge_node")
    graph.add_edge("crossref_node", "merge_node")
    graph.add_edge("merge_node", "filter_node")
    graph.add_edge("filter_node", "save_node")
    graph.add_edge("save_node", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({
        # Add / edit queries here for broader dataset coverage.
        # Each query is run against all 3 sources (arXiv, PubMed, CrossRef).
        "queries": [
            "IoT healthcare monitoring",
            "IoMT wearable sensors",
            "remote patient monitoring IoT",
            "smart healthcare devices machine learning",
        ],
        "max_results_per_source": 15,  # per query, per source -> up to 4*3*15=180 raw results before dedupe
        "all_papers": [],
        "merged_papers": [],
        "filtered_papers": [],
    })
    print(f"\nFinal merged count (pre-filter): {len(result['merged_papers'])}")
    print(f"Final filtered count (has abstract): {len(result['filtered_papers'])}")
