import logging
import feedparser
from datetime import datetime, timedelta, timezone
from typing import List
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from llm import get_llm
from persistence import load_seen_urls

llm = get_llm()


def run_arxiv_agent(target: int = 25) -> list:
    seen_urls = load_seen_urls()
    paper_registry = {}   # arxiv_id -> {title, url, abstract}
    result = {"papers": []}

    @tool
    def search_arxiv(query: str, max_results: int = 20) -> str:
        """Search arXiv for recent papers using the arXiv API query syntax.
        Supports field prefixes: ti: (title), abs: (abstract), au: (author).
        Supports boolean operators: AND, OR, ANDNOT.
        To filter by date: append AND submittedDate:[YYYYMMDD TO YYYYMMDD].
        max_results is capped at 30. Returns paper IDs, titles and abstracts."""
        url = (
            f"http://export.arxiv.org/api/query"
            f"?search_query={query}"
            f"&max_results={min(max_results, 30)}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )
        feed = feedparser.parse(url)
        new_papers = []
        for entry in feed.entries:
            arxiv_id = entry.id.split("/abs/")[-1]
            paper_url = f"https://arxiv.org/abs/{arxiv_id}"
            if paper_url in seen_urls:
                continue
            if arxiv_id not in paper_registry:
                paper_registry[arxiv_id] = {
                    "title": entry.title.replace("\n", " ").strip(),
                    "url": paper_url,
                    "abstract": entry.summary.replace("\n", " ").strip(),
                }
            new_papers.append(arxiv_id)

        if not new_papers:
            return "No new papers found for this query."

        lines = []
        for arxiv_id in new_papers:
            p = paper_registry[arxiv_id]
            lines.append(
                f"ID: {arxiv_id}\n"
                f"Title: {p['title']}\n"
                f"Abstract: {p['abstract'][:400]}..."
            )
        return f"Found {len(new_papers)} new papers:\n\n" + "\n\n---\n\n".join(lines)

    @tool
    def get_abstract(arxiv_id: str) -> str:
        """Get the full abstract of a paper by its arXiv ID (e.g. '2306.00978').
        Use this when a title is ambiguous and you need more context to judge relevance."""
        if arxiv_id in paper_registry:
            p = paper_registry[arxiv_id]
            return f"Title: {p['title']}\nAbstract: {p['abstract']}"
        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        feed = feedparser.parse(url)
        if not feed.entries:
            return "Paper not found."
        entry = feed.entries[0]
        paper_url = f"https://arxiv.org/abs/{arxiv_id}"
        paper_registry[arxiv_id] = {
            "title": entry.title.replace("\n", " ").strip(),
            "url": paper_url,
            "abstract": entry.summary.replace("\n", " ").strip(),
        }
        return f"Title: {paper_registry[arxiv_id]['title']}\nAbstract: {paper_registry[arxiv_id]['abstract']}"

    @tool
    def finalize(arxiv_ids: List[str], explanations: List[str]) -> str:
        """Submit your final paper selection.
        arxiv_ids: list of selected arXiv paper IDs.
        explanations: 2-3 sentence plain-English explanation for each paper
                      (same order as arxiv_ids), explaining what it contributes
                      and why it matters for someone learning LLM inference.
        Call this when you have selected 20-30 papers."""
        papers = []
        for arxiv_id, explanation in zip(arxiv_ids, explanations):
            if arxiv_id in paper_registry:
                p = paper_registry[arxiv_id]
                papers.append({
                    "arxiv_id": arxiv_id,
                    "title": p["title"],
                    "url": p["url"],
                    "explanation": explanation,
                })
        result["papers"] = papers
        return f"Finalized {len(papers)} papers."

    agent = create_react_agent(llm, [search_arxiv, get_abstract, finalize])

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=7)
    date_range = f"{start_date.strftime('%Y%m%d')} TO {end_date.strftime('%Y%m%d')}"

    prompt = (
        f"You are an AI research curator helping a developer deeply learn about LLM systems.\n\n"
        f"Your goal: find {target} of the most important arXiv papers from the past 7 days "
        f"(date range: {date_range}) on these 4 topics:\n"
        "1. LLM inference optimization (KV cache, speculative decoding, batching, memory, serving systems)\n"
        "2. Model quantization and compression (weight quantization, activation quantization, pruning, distillation)\n"
        "3. New transformer architectures (attention variants, MoE, SSM/Mamba, linear attention)\n"
        "4. New model releases and benchmarks\n\n"
        "Instructions:\n"
        "- Use search_arxiv with targeted queries. Include date range in queries when possible.\n"
        "- Use get_abstract when a title is unclear before deciding.\n"
        "- Prefer papers with concrete, measurable contributions: numerical improvements "
        "  (e.g. 2x speedup, 50% memory reduction), new model/code releases, or novel "
        "  techniques that address a known practical limitation.\n"
        "- Deprioritize incremental improvements with marginal gains, pure theoretical "
        "  analysis without experiments, or workshop papers.\n"
        "- If a paper announces an open-source release or benchmark, prioritize it.\n"
        "- As a tiebreaker between equally strong papers, prefer those from well-known labs "
        "  (Google, Meta, Microsoft, NVIDIA, Anthropic, OpenAI, CMU, Stanford, MIT, "
        "  UC Berkeley, Tsinghua, PKU).\n"
        "- Ensure balanced coverage: at least 5 papers per topic.\n"
        "- When you have 20-30 high-quality papers, call finalize.\n"
        "- Each explanation must be 2-3 sentences: what the paper does and why it matters."
    )

    logging.info("[arxiv_agent] starting research agent...")
    agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    logging.info(f"[arxiv_agent] agent finished, selected {len(result['papers'])} papers")

    return result["papers"]
