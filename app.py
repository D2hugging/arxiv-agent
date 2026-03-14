import os
import logging
import requests
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from arxiv_agent import run_arxiv_agent
from formatter import format_arxiv_markdown
from persistence import save_seen_urls

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)

app = FastAPI(title="arXiv Agent")


def send_to_discord(markdown: str):
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        logging.warning("DISCORD_WEBHOOK_URL not set, skipping.")
        return
    parts = markdown.split("\n")
    header = parts[0]
    message_chunk = header + "\n"
    for line in parts[1:]:
        if len(message_chunk) + len(line) + 1 > 2000:
            requests.post(webhook_url, json={"content": message_chunk})
            message_chunk = ""
        message_chunk += line + "\n"
    if message_chunk:
        requests.post(webhook_url, json={"content": message_chunk})


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
        send_to_discord(markdown)
        save_seen_urls({paper["url"] for paper in papers})
        logging.info("Done.")

    background_tasks.add_task(run_once)
    return JSONResponse({"status": "accepted", "message": "arXiv agent running in background."})


@app.get("/api/ping")
def ping():
    return JSONResponse({"status": "ok", "app": "arXiv Agent"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
