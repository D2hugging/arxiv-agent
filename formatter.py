def format_arxiv_markdown(papers: list) -> str:
    lines = ["### Weekly arXiv: LLM Inference, Quantization & Architecture\n"]
    for i, paper in enumerate(papers, 1):
        lines.append(f"{i}. **[{paper['title']}]({paper['url']})**")
        lines.append(f"   {paper['explanation']}\n")
    return "\n".join(lines)
