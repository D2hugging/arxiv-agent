import json
import logging
import os
from datetime import datetime, timezone, timedelta
from huggingface_hub import hf_hub_download, upload_file

HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")
HF_TOKEN = os.getenv("HF_TOKEN")
SEEN_FILE = "arxiv_seen_urls.json"   # {url: datetime} — rolling 180 days
RECENT_DAYS = 180


def _download():
    try:
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=SEEN_FILE,
            repo_type="dataset",
            force_download=True,
            token=HF_TOKEN,
        )
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"[persistence] failed to download {SEEN_FILE}: {e}")
        return {}


def _upload(data):
    try:
        upload_file(
            path_or_fileobj=json.dumps(data, ensure_ascii=False, indent=2).encode(),
            path_in_repo=SEEN_FILE,
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
            commit_message=f"update {SEEN_FILE}",
        )
        logging.info(f"[persistence] uploaded {SEEN_FILE}")
    except Exception as e:
        logging.error(f"[persistence] failed to upload {SEEN_FILE}: {e}")


def load_seen_urls() -> set:
    if not HF_DATASET_REPO:
        logging.warning("[persistence] HF_DATASET_REPO not set, dedup skipped")
        return set()
    seen = _download()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=RECENT_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    result = {url for url, dt in seen.items() if dt >= cutoff}
    logging.info(f"[persistence] loaded {len(result)} seen URLs")
    return result


def save_seen_urls(urls: set):
    if not HF_DATASET_REPO:
        logging.warning("[persistence] HF_DATASET_REPO not set, URLs not saved")
        return
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    seen = _download()
    for url in urls:
        if url not in seen:
            seen[url] = now
    cutoff = (datetime.now(timezone.utc) - timedelta(days=RECENT_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    seen = {url: dt for url, dt in seen.items() if dt >= cutoff}
    _upload(seen)
