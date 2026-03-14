import os

LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not LOCAL_LLM_URL and not GOOGLE_API_KEY:
    raise RuntimeError("please set GOOGLE_API_KEY or LOCAL_LLM_URL")
