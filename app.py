import os
import time
import logging
import requests
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from arxiv_agent import run_arxiv_agent
from formatter import format_arxiv_markdown
from persistence import save_seen_urls, save_selected_papers

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)

app = FastAPI(title="arXiv Agent")


def send_to_discord(markdown: str, retries: int = 3) -> bool:
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        logging.warning("DISCORD_WEBHOOK_URL not set, skipping.")
        return False
    parts = markdown.split("\n")
    header = parts[0]
    chunks = []
    message_chunk = header + "\n"
    for line in parts[1:]:
        if len(message_chunk) + len(line) + 1 > 2000:
            chunks.append(message_chunk)
            message_chunk = header + "\n"
        message_chunk += line + "\n"
    if message_chunk:
        chunks.append(message_chunk)

    for attempt in range(retries):
        try:
            for chunk in chunks:
                requests.post(webhook_url, json={"content": chunk}).raise_for_status()
            return True
        except Exception as e:
            logging.warning(f"[discord] attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    logging.error("[discord] all retry attempts failed.")
    return False


@app.get("/")
def root():
    return JSONResponse({"status": "ok", "endpoints": ["/v1/arxiv/fetch", "/api/ping"]})


@app.post("/v1/arxiv/fetch")
async def fetch_arxiv_endpoint(request: Request, background_tasks: BackgroundTasks):
    logging.info(f"API /v1/arxiv/fetch called from {request.client.host}")

    secret_token = os.getenv("HF_TOKEN")
    auth_header = request.headers.get("Authorization")
    if secret_token and auth_header != f"Bearer {secret_token}":
        logging.warning("Unauthorized access attempt.")
        raise HTTPException(status_code=403, detail="Forbidden: Invalid or missing token.")

    def run_once():
        logging.info("Running arxiv agent...")
        papers = run_arxiv_agent()
        if not papers:
            logging.warning("Agent returned no papers.")
            return
        markdown = format_arxiv_markdown(papers)
        logging.info(f"Formatted {len(papers)} papers.")
        save_selected_papers(papers)
        sent = send_to_discord(markdown)
        if sent:
            save_seen_urls({paper["url"] for paper in papers})
            logging.info("Done.")
        else:
            logging.warning("Discord send failed after retries. Papers saved to selected log but URLs not marked as seen.")

    background_tasks.add_task(run_once)
    return JSONResponse({"status": "accepted", "message": "arXiv agent running in background."})


@app.get("/api/ping")
def ping():
    return JSONResponse({"status": "ok", "app": "arXiv Agent"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
