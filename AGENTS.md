# AGENTS.md

## Project Overview

**LocalBrain** - A privacy-first local AI assistant that indexes all your personal data (AI conversations, notes, documents) and lets you chat with it locally.

Combines:
- **pi-brain** (https://github.com/0xSero/pi-brain) - Extracts AI coding sessions with privacy redaction
- **RAG engine** - BM25 retrieval, embeddings, web UI, workflows

## MVP Features

1. **Chat with your data** - Web UI + CLI to query all your AI conversations, notes, documents
2. **Knowledge Indexing** - Ingest from multiple sources (Pi, Claude, Codex, Gemini, Factory, Qwen Code, etc.)

All local, all private.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    LOCALBRAIN v2                     │
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
│    (smart chunking: auto/fixed/sliding/recursive/    │
│     semantic/conversation-aware)                    │
│         │                                           │
│         ▼                                           │
│  SQLite + sqlite-vec (embeddings)                   │
│    data/state.db                                    │
│         │                                           │
│    ┌────┴────┐                                      │
│    ▼         ▼                                      │
│  RETRIEVAL   OUTPUTS                                │
│  ┌─────────┐ • Web UI (chat interface)              │
│  │Query    │ • Morning Briefing                     │
│  │Planner  │ • arXiv Queue                          │
│  └────┬────┘ • Quality Metrics                      │
│       ▼                                             │
│  Hybrid Retrieval:                                  │
│   1. BM25 (FTS5) → broad keyword recall            │
│   2. Vector (sqlite-vec) → semantic similarity      │
│   3. Merge + deduplicate                            │
│   4. Cross-encoder rerank (optional)                │
│   5. Recency + importance scoring                   │
│       ▼                                             │
│  LOCAL LLM (llama.cpp, any GPU)                     │
│  - Adapts prompt to query type                     │
│  - Comparative queries get more context             │
│  - Quality-tracked responses                        │
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
python scripts/ingest/chunk.py          # Smart chunking (auto strategy)
python scripts/ingest/embedding.py     # Real sentence-transformer embeddings

# Start web UI (chat interface with query planning)
python scripts/web_server.py

# Query knowledge base
python scripts/query.py "search term" --db data/state.db
python scripts/query.py --recent 7 --db data/state.db

# Hybrid retrieval (direct)
python scripts/core/retriever.py "search term" --strategy auto --rerank

# Query planning (analyze optimal strategy)
python scripts/core/query_planner.py "what hardware do I have?"

# Quality metrics
python scripts/core/quality.py --report --index-stats
python scripts/core/quality.py --failures

# Smart chunking (test strategies)
python scripts/core/smart_chunk.py --strategy auto --text "your text here"
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
├── schemas/          # SQL schema definitions
├── scripts/          # Python pipeline scripts
│   ├── core/         # Shared utilities
│   │   ├── retriever.py      # Hybrid retrieval (BM25 + Vector + Rerank)
│   │   ├── query_planner.py  # Auto-selects best retrieval strategy
│   │   ├── smart_chunk.py    # Strategy-aware chunking
│   │   ├── quality.py        # Retrieval quality metrics
│   │   ├── hybrid_search.py  # Legacy hybrid search
│   │   ├── bm25.py           # BM25/FTS5 search
│   │   └── llm_client.py     # LLM client
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

## Retrieval Architecture (v2)

Based on analysis of RAG vs Long Context approaches:

### Query Planning
Queries are auto-classified into types, each with optimal retrieval:

| Query Type | Strategy | Why |
|------------|----------|-----|
| Factual | BM25 only | Keywords are precise |
| Comparative | Long context | Solves "whole book problem" |
| Analytical | Hybrid + rerank | Needs broad context with precision |
| Temporal | Hybrid + recency | Fresh results matter |
| Personal | Hybrid | May be in profile or knowledge base |
| Code | BM25 only | Code has specific keywords |
| Exploratory | Hybrid | Broad search with semantic recall |

### Retrieval Pipeline
1. **BM25 (FTS5)** → fast keyword candidates (broad recall)
2. **Vector search (sqlite-vec)** → semantic similarity
3. **Merge + deduplicate** → combine both signals, boost mutual hits
4. **Cross-encoder rerank** → precision boost (optional)
5. **Recency + importance** → final scoring

### Smart Chunking
Auto-selects strategy based on content type:
- **Conversations** → turn-aware (groups complete speaker turns)
- **Markdown/docs** → semantic (groups by headings/paragraphs)
- **Code** → fixed-size (predictable boundaries)
- **General text** → recursive (respects paragraph → sentence → char)
- **Sliding window** → overlap for context continuity

### Quality Metrics
Tracks retrieval performance to detect:
- Zero-result queries (retrieval failures)
- Low-confidence results (potential silent failures)
- Latency outliers
- Source diversity

## What Makes This Unique

| Feature | LocalBrain v2 | RAGFlow | Others |
|---------|---------------|---------|--------|
| AI conversation extraction | Yes (pi-brain) | No | No |
| Privacy redaction | Yes (built-in) | Partial | Varies |
| No Docker required | Yes | No | Varies |
| Multi-platform | Yes (GPU + CPU) | Optional | Varies |
| Open source | Yes | Yes | Varies |
| SQLite-only | Yes | ES/MySQL | Varies |
| Query planning | Yes | No | Rare |
| Smart chunking | Yes (5 strategies) | Basic | Basic |
| Hybrid retrieval | BM25+Vector+Rerank | Varies | Varies |
| Quality tracking | Yes | No | No |
| Silent failure detection | Yes | No | No |

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
