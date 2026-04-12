# AGENTS.md

## Project Overview

**LocalBrain** - A privacy-first local AI assistant that indexes all your personal data (AI conversations, notes, documents) and lets you chat with it locally.

Combines:
- **pi-brain** (https://github.com/0xSero/pi-brain) - Extracts AI coding sessions with privacy redaction
- **RAG engine** - BM25 retrieval, embeddings, web UI, workflows

## MVP Features

1. **Chat with your data** - Web UI + CLI to query all your AI conversations, notes, documents
2. **Knowledge Indexing** - Ingest from multiple sources (Pi, Claude, Codex, Gemini, Factory, Qwen Code, etc.)
3. **Morning Briefing** - Daily digest of important items and pending actions
4. **arXiv Workflow** - Fetch, score, and triage AI research papers

All local, all private.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    LOCALBRAIN                        │
├─────────────────────────────────────────────────────┤
│  DATA SOURCES                                       │
│    Pi sessions         ~/.pi/agent/sessions/        │
│    Claude Code         ~/.claude/projects/          │
│    Codex               ~/.codex/sessions/           │
│    OpenCode            ~/.local/share/opencode/     │
│    Cursor              ~/extracted_data/            │
│    Factory (DROID)     ~/.factory/sessions/         │
│    Qwen Code           ~/.qwen/projects/*/chats/    │
│    Gemini Takeout      exports/takeout-*.zip        │
│         │                                           │
│         ▼                                           │
│  INGESTION PIPELINE                                 │
│    discover → classify → extract → chunk → embed    │
│         │                                           │
│         ▼                                           │
│  SQLite + sqlite-vec (embeddings)                   │
│    data/state.db                                    │
│         │                                           │
│    ┌────┴────┐                                      │
│    ▼         ▼                                      │
│  LOCAL LLM   OUTPUTS                                │
│  llama.cpp   • Web UI (chat interface)              │
│  (any GPU)   • Morning Briefing                     │
│              • arXiv Queue                          │
└─────────────────────────────────────────────────────┘
```

## Quick Reference

### Commands
```bash
# Initialize database
python scripts/ops/init_db.py

# Run full ingestion
python scripts/ingest/discover.py
python scripts/ingest/classify.py
python scripts/ingest/extract.py
python scripts/ingest/chunk.py
python scripts/ingest/embedding.py

# Start web UI (chat interface)
python scripts/web_server.py

# Generate morning briefing
python scripts/briefing/build_briefing.py

# Query knowledge base
python scripts/query.py "search term" --db data/state.db
python scripts/query.py --recent 7 --db data/state.db
```

### Data Source Setup

#### 1. Extract coding sessions with pi-brain
```bash
npm install -g @0xsero/pi-brain

pi-brain export pi --output exports/pi_sessions.jsonl
pi-brain export claude --output exports/claude_sessions.jsonl
pi-brain export codex --output exports/codex_sessions.jsonl
```

#### 2. Factory (DROID) Sessions
```bash
# Auto-extracted from ~/.factory/sessions/
python scripts/ingest/factory_extractor.py
```

#### 3. Qwen Code Sessions
```bash
# Auto-extracted from ~/.qwen/projects/
python scripts/ingest/qwen_extractor.py
```

#### 4. Gemini Takeout
1. Download from https://takeout.google.com/settings/takeout (select "Gemini")
2. Place ZIP in `exports/`
3. Extract and normalize:
```bash
python parse_gemini_html.py
python normalize_gemini.py
```

#### 5. Run ingestion
```bash
python scripts/ingest/discover.py
python scripts/ingest/classify.py
python scripts/ingest/extract.py
python scripts/ingest/chunk.py
python scripts/ingest/embedding.py
```

## Hardware

Supports NVIDIA GPU, AMD GPU, Apple Silicon (Metal), and CPU-only.

- **NVIDIA GPU**: Full CUDA offload via llama.cpp — set `gpu_layers: 99`
- **Apple Silicon**: Metal auto-detected — set `gpu_layers: 99`
- **CPU-only**: Works fine — set `gpu_layers: 0`
- **Storage**: Local SSD for SQLite database + embeddings

See `config/models.yaml` for model and hardware settings.

## Directory Structure

```
localbrain/
├── config/           # YAML configs (models, sources, scoring)
├── data/             # SQLite database, logs, manifests
├── exports/          # Raw data sources (takeout ZIPs, JSONL)
├── prompts/          # LLM prompt templates
├── scripts/          # Python pipeline scripts
│   ├── core/         # Shared utilities (llm_client, paths, bm25)
│   ├── ingest/       # Ingestion pipeline
│   └── ops/          # Database operations
├── runtime/          # Web UI (index.html, app.js, style.css)
├── models/           # GGUF model files (download separately)
└── AGENTS.md         # This file
```

## Project Goals

1. **Privacy-first** - All data stays local, no cloud APIs
2. **Efficient RAG** - BM25 + embeddings, hybrid retrieval
3. **Chat interface** - Web UI to query all personal data
4. **Workflow automation** - Briefings, arXiv triage
5. **Open source** - Ship to GitHub as usable MVP

## What Makes This Unique

| Feature | LocalBrain | RAGFlow | Others |
|---------|------------|---------|--------|
| AI conversation extraction | Yes (pi-brain) | No | No |
| Privacy redaction | Yes (built-in) | Partial | Varies |
| No Docker required | Yes | No | Varies |
| Multi-platform | Yes (GPU + CPU) | Optional | Varies |
| Personal workflows | Yes | Generic | Generic |
| SQLite-only | Yes | ES/MySQL | Varies |

## Rules

- All LLM inference runs locally via llama.cpp
- No data sent to external APIs (except optional arXiv fetch)
- Keep scripts simple and maintainable
- Preserve provenance for every chunk

## Recommended Models

| Model | Size (Q4_K_M) | Min VRAM | Best For |
|-------|---------------|----------|----------|
| **Gemma 4 12B** | ~8 GB | 10 GB | Best overall — fast, accurate, efficient |
| **Qwen 3.5 4B** | ~3 GB | 4 GB | Lightest — low-VRAM or testing |
| **Qwen 3.5 9B** | ~6 GB | 8 GB | Sweet spot — quality vs speed |
| **Qwen 3.5 27B** | ~17 GB | 20 GB | Best quality — deep reasoning |
| **Qwen 3.5 60B** | ~36 GB | 40 GB | Maximum — complex analysis |
| **Gemma 4 48B** | ~30 GB | 36 GB | Large-scale reasoning, creative writing |
| **Llama 3.3 8B** | ~5 GB | 6 GB | General purpose, widely supported |
| **Mistral Small 3.1** | ~14 GB | 16 GB | Efficient coding and tool use |

Download GGUF quants from [Hugging Face](https://huggingface.co/models?library=gguf&sort=trending).

**Quick start:** `Qwen 3.5 4B` for testing, `Gemma 4 12B` for daily use, `Qwen 3.5 27B` for best quality. Use Q4_K_M quantization — near-lossless at half the size.

## Getting Started

1. Clone repo
2. Install dependencies: `pip install -r requirements.txt`
3. Download a GGUF model to `models/`
4. Place your data exports in `exports/`
5. Run: `python scripts/ops/init_db.py`
6. Ingest: `python scripts/ingest/discover.py` etc.
7. Chat: `python scripts/web_server.py`
8. Open: http://localhost:8096
