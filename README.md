---
title: arXiv Agent
emoji: 📄
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.35.0
app_file: app.py
pinned: false
---

# arXiv Agent

A weekly arXiv paper digest agent for LLM inference, quantization, and architecture research.

## Endpoints

- `POST /v1/arxiv/fetch` — trigger the agent (requires `Authorization: Bearer <HF_TOKEN>`)
- `GET /api/ping` — health check
