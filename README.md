<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey?style=for-the-badge" alt="Platform">
  <img src="https://img.shields.io/badge/LLM-llama.cpp-orange?style=for-the-badge" alt="llama.cpp">
  <br>
  <img src="https://img.shields.io/badge/Sources-10+-purple?style=flat-square" alt="10+ Sources">
  <img src="https://img.shields.io/badge/Retrieval-Hybrid%20(BM25+%20Reranker)-blue?style=flat-square" alt="Hybrid Retrieval">
  <img src="https://img.shields.io/badge/Private-100%25-success?style=flat-square" alt="100% Private">
</p>

<h1 align="center">&#x1F9E0; LocalBrain</h1>
<p align="center"><strong>Chat with your entire AI history.</strong></p>
<p align="center">
  Index conversations from Pi, Claude Code, Codex, DROID/Factory, Qwen Code, Hermes, OpenClaw, Gemini &amp; more —
  then query them with a local LLM. Zero cloud. All yours.
</p>

<p align="center">
  <a href="#quick-start"><strong>Quick Start</strong></a> &#183;
  <a href="#supported-sources"><strong>Sources</strong></a> &#183;
  <a href="#recommended-models"><strong>Models</strong></a> &#183;
  <a href="#architecture"><strong>Architecture</strong></a> &#183;
  <a href="#data-extraction"><strong>Extract Data</strong></a>
</p>

---

## &#x2728; What Is This?

LocalBrain is a **privacy-first local AI assistant** that indexes all your AI conversations — from coding sessions (Pi, Claude, Codex, Hermes, OpenClaw, Factory/DROID, Qwen Code), chat takeouts (Gemini), and notes — then lets you **chat with your entire history** using GPU-accelerated local inference.

> Think of it as a personal second brain that actually remembers everything.

## &#x2728; Features

| Feature | Details |
|---------|---------|
| **10+ sources** | Pi, Claude, Codex, OpenCode, Cursor, Factory, Hermes, OpenClaw, Qwen Code, Gemini |
| **Hybrid retrieval** | BM25 keyword search + cross-encoder reranking (ms-marco-MiniLM-L-6-v2) |
| **Local LLM** | Qwen 3.5, Gemma 4, Llama 3.3, Mistral — any GGUF via llama.cpp |
| **Web UI** | Clean chat interface with source citations |
| **Personal profile** | Edit `runtime/personal.json` for contextual answers about you |
| **100% private** | No cloud APIs, no data leaves your machine, no vector DB, MIT licensed |

## &#x1F680; Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Initialize database
python scripts/ops/init_db.py

# 3. Place your data exports in exports/
#    (see Data Extraction section below)

# 4. Run full ingestion pipeline
python scripts/ingest/discover.py
python scripts/ingest/classify.py
python scripts/ingest/extract.py
python scripts/ingest/chunk.py

# 5. Start llama-server with a GGUF model
llama-server -m models/qwen3.5-4b.gguf --port 8080 -ngl 99 -c 8192

# 6. Start web UI -> open http://localhost:8096
python scripts/web_server.py

# Or query from CLI
python scripts/query.py "What have I worked on?" --db data/state.db
```

## &#x1F4E6; Supported Sources

| Source | Extraction | Status |
|--------|-----------|--------|
| **[Pi](https://pi.ai)** | `pi-brain export pi` | &#x2705; |
| **[Claude Code](https://claude.ai/code)** | `pi-brain export claude` | &#x2705; |
| **[Codex](https://openai.com/codex)** | `pi-brain export codex` | &#x2705; |
| **[OpenCode](https://github.com/opencode-ai/opencode)** | Auto-discovered | &#x2705; |
| **[Cursor](https://cursor.sh)** | Pre-extracted JSONL | &#x2705; |
| **[Factory / DROID](https://github.com/0xSero/pi-brain)** | Auto-discovered | &#x2705; |
| **[Hermes](https://github.com/openclaw/hermes)** | `pi-brain export hermes` | &#x2705; |
| **[OpenClaw](https://github.com/openclaw)** | `pi-brain export openclaw` | &#x2705; |
| **[Qwen Code](https://github.com/QwenLM)** | Auto-discovered | &#x2705; |
| **[Google Gemini](https://gemini.google.com)** | Takeout HTML parser | &#x2705; |

## &#x1F916; Recommended Models

| Model | Size (Q4_K_M) | Min VRAM | Best For |
|-------|---------------|----------|----------|
| **Gemma 4 12B** | ~8 GB | 10 GB | &#x2B50; Best overall |
| **Qwen 3.5 4B** | ~3 GB | 4 GB | &#x1F680; Lightest / testing |
| **Qwen 3.5 9B** | ~6 GB | 8 GB | &#x1F3AF; Sweet spot |
| **Qwen 3.5 27B** | ~17 GB | 20 GB | &#x1F9E0; Best quality |
| **Qwen 3.5 60B** | ~36 GB | 40 GB | &#x1F52C; Maximum reasoning |
| **Gemma 4 48B** | ~30 GB | 36 GB | &#x270D;&#xFE0F; Creative writing |
| **Llama 3.3 8B** | ~5 GB | 6 GB | &#x1F310; General purpose |
| **Mistral Small 3.1** | ~14 GB | 16 GB | &#x1F4BB; Coding |

> **Quick pick:** `Qwen 3.5 4B` for testing, `Gemma 4 12B` for daily use, `Qwen 3.5 27B` for best quality.
> Download GGUF quants from [Hugging Face](https://huggingface.co/models?library=gguf&sort=trending).

### CPU-Only Mode

No GPU? llama-server runs on CPU too — set `gpu_layers: 0` in `config/models.yaml`. Expect ~10-20 tok/s on a modern CPU with 8B models.

### Apple Silicon

On Mac with M-series chips, llama.cpp auto-detects Metal GPU. Set `gpu_layers: 99` for full offload.

## &#x1F3D7; Architecture

```
+--------------------------------------------------------------+
|                        LOCALBRAIN                             |
+--------------------------------------------------------------+
|  DATA SOURCES                                                 |
|  Pi . Claude . Codex . OpenCode . Cursor . Factory            |
|  Hermes . OpenClaw . Qwen Code . Gemini Takeout               |
+---------------------------+----------------------------------+
|                           v                                  |
|  INGESTION PIPELINE                                           |
|  discover -> classify -> extract -> chunk -> index            |
+---------------------------+----------------------------------+
|                           v                                  |
|  SQLite (content_chunks) + BM25 in-memory index               |
+---------------+-------------------+--------------------------+
|               v                   v                          |
|  +-----------+-----+     +--------+-----------+              |
|  |  Web UI (:8096) |     |  CLI Query         |              |
|  +-----------+-----+     +--------+-----------+              |
|               |                   |                          |
|               +--------+----------+                          |
|                        v                                     |
|  +----------------------------------------------------+      |
|  |  HYBRID RETRIEVAL                                    |      |
|  |  1. BM25 -> top 50 candidates (keyword match)       |      |
|  |  2. Cross-encoder -> rerank (semantic scoring)      |      |
|  |  3. Return top 10                                   |      |
|  +----------------------------------------------------+      |
|                        v                                     |
|  +----------------------------------------------------+      |
|  |  llama-server (:8080)                               |      |
|  |  Qwen 3.5 / Gemma 4 / Llama 3.3 / Any GGUF          |      |
|  |  Full GPU offload . Apple Silicon . CPU fallback      |      |
|  +----------------------------------------------------+      |
+--------------------------------------------------------------+
```

### Why Hybrid Retrieval (Not Vector Embeddings)?

| Approach | Storage | Speed | Accuracy | Complexity |
|----------|---------|-------|----------|------------|
| BM25 only | None | Fastest | Good | Simple |
| **Hybrid (BM25 + reranker)** | None | Fast | **Best** | Moderate |
| Vector embeddings | 622K x 768d (~2GB) | Medium | Good | Complex |

Hybrid retrieval is what production RAG systems actually use. The cross-encoder sees the query and document **together** (not as independent embeddings), giving much more accurate relevance scores.

## &#x1F4C2; Data Extraction

### From pi-brain (Pi, Claude, Codex, OpenCode, Cursor, Hermes, OpenClaw)

```bash
npm install -g @0xsero/pi-brain

pi-brain export pi --output exports/pi_sessions.jsonl
pi-brain export claude --output exports/claude_sessions.jsonl
pi-brain export codex --output exports/codex_sessions.jsonl
pi-brain export hermes --output exports/hermes_sessions.jsonl
```

**Enhanced pi-brain fork** with Factory/DROID, Gemini Takeout & Qwen Code plugins:
[github.com/0xSero/pi-brain](https://github.com/0xSero/pi-brain)

### From Factory / DROID

Auto-discovered from `~/.factory/sessions/`:

```bash
python scripts/ingest/factory_extractor.py
```

### From Qwen Code

Auto-discovered from `~/.qwen/projects/*/chats/`:

```bash
python scripts/ingest/qwen_extractor.py
```

### From Google Gemini Takeout

1. Download from [Google Takeout](https://takeout.google.com/settings/takeout) -> select **"Gemini"**
2. Place ZIP(s) in `exports/`
3. Extract and normalize:

```bash
python parse_gemini_html.py    # Streaming parser for 60MB+ HTML files
python normalize_gemini.py      # Converts to ingestable JSONL
```

## &#x2699; Configuration

### Personal Profile

Edit `runtime/personal.json` so the AI knows about you:

```json
{
  "name": "Your Name",
  "location": "Your Location",
  "hardware": { "gpu": "Your GPU", "ram_system": "Your RAM" },
  "tools_used": ["Pi", "Claude Code", "Codex", "Gemini"],
  "projects": ["Project 1", "Project 2"],
  "interests": ["AI", "coding"]
}
```

### Model Settings

Edit `config/models.yaml` for model paths, GPU layers, and context size.

## &#x1F9EA; Benchmark

```bash
python benchmark.py
```

Runs 20 diverse questions scoring retrieval accuracy + LLM answer quality.
Typical score: **94%** (75/80).

## &#x1F4C1; Project Structure

```
localbrain/
├── config/             # YAML configs (models, sources)
├── data/               # SQLite database
├── exports/            # Raw data sources (takeout ZIPs, JSONL)
├── runtime/            # Web UI (index.html, app.js, style.css)
├── scripts/
│   ├── core/           # Hybrid search (BM25 + cross-encoder), LLM client
│   ├── ingest/         # Pipeline (discover->classify->extract->chunk)
│   └── ops/            # Database operations
├── models/             # GGUF model files (download separately)
├── parse_gemini_html.py  # Google Takeout HTML parser
└── benchmark.py          # Quality test suite
```

## &#x1F512; Privacy Guarantees

- **100% local** — no data ever leaves your machine
- **No cloud APIs** — inference runs via llama.cpp on your hardware
- **No vector database** — BM25 + cross-encoder, zero storage overhead
- **SQLite only** — no external databases or services
- **Open source** — audit everything, MIT licensed

## &#x1F91D; Related Projects

| Project | Description |
|---------|-------------|
| [pi-brain](https://github.com/0xSero/pi-brain) | Original session extractor — Pi, Claude, Codex exports |
| [SergiioB/pi-brain](https://github.com/SergiioB/pi-brain) | Enhanced fork with Factory/DROID, Hermes, OpenClaw, Gemini Takeout & Qwen Code plugins |

## &#x1F4DC; License

MIT — do whatever you want with it.
