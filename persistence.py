import json
import logging
import os
from datetime import datetime, timezone, timedelta
from huggingface_hub import hf_hub_download, upload_file

HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")
HF_TOKEN = os.getenv("HF_TOKEN")
SEEN_FILE = "arxiv_seen_urls.json"       # {url: datetime} — rolling 180 days
SELECTED_FILE = "arxiv_selected_papers.json"  # [{run_date, papers}] — all runs
RECENT_DAYS = 180


def _download(filename: str):
    try:
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=filename,
            repo_type="dataset",
            force_download=True,
            token=HF_TOKEN,
        )
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"[persistence] failed to download {filename}: {e}")
        return None


def _upload(filename: str, data):
    try:
        upload_file(
            path_or_fileobj=json.dumps(data, ensure_ascii=False, indent=2).encode(),
            path_in_repo=filename,
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
            commit_message=f"update {filename}",
        )
        logging.info(f"[persistence] uploaded {filename}")
    except Exception as e:
        logging.error(f"[persistence] failed to upload {filename}: {e}")


def load_seen_urls() -> set:
    if not HF_DATASET_REPO:
        logging.warning("[persistence] HF_DATASET_REPO not set, dedup skipped")
        return set()
    seen = _download(SEEN_FILE) or {}
    cutoff = (datetime.now(timezone.utc) - timedelta(days=RECENT_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    result = {url for url, dt in seen.items() if dt >= cutoff}
    logging.info(f"[persistence] loaded {len(result)} seen URLs")
    return result


def save_seen_urls(urls: set):
    if not HF_DATASET_REPO:
        logging.warning("[persistence] HF_DATASET_REPO not set, URLs not saved")
        return
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    seen = _download(SEEN_FILE) or {}
    for url in urls:
        if url not in seen:
            seen[url] = now
    cutoff = (datetime.now(timezone.utc) - timedelta(days=RECENT_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    seen = {url: dt for url, dt in seen.items() if dt >= cutoff}
    _upload(SEEN_FILE, seen)


def save_selected_papers(papers: list):
    if not HF_DATASET_REPO:
        logging.warning("[persistence] HF_DATASET_REPO not set, selected papers not saved")
        return
    existing = _download(SELECTED_FILE)
    if not isinstance(existing, list):
        existing = []
    existing.append({
        "run_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "papers": papers,
    })
    _upload(SELECTED_FILE, existing)
