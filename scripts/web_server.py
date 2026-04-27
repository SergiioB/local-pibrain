#!/usr/bin/env python3
"""
LocalBrain - Web server with intelligent hybrid RAG + LLM.

Improvements from "Is RAG Still Needed?" analysis:
  1. Query planning: auto-selects best retrieval strategy per query
  2. Hybrid retrieval: BM25 + vector + cross-encoder reranking
  3. Quality tracking: monitors retrieval performance
  4. Context management: adapts context size to query type
  5. Better UI: shows strategy, scores, provenance
"""
import json
import os
import sqlite3
import sys
import time
import urllib.request
from http.server import SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from http.server import HTTPServer
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RUNTIME_DIR = PROJECT_ROOT / "runtime"
DB_PATH = DATA_DIR / "state.db"
PORT = int(os.environ.get("PERSONAL_AI_WEB_PORT", "8096"))
HOST = "127.0.0.1"

sys.path.insert(0, str(SCRIPT_DIR))

# --- Personal Profile ---
PERSONAL_PATH = RUNTIME_DIR / "personal.json"
_personal = {}
if PERSONAL_PATH.exists():
    try:
        _personal = json.loads(PERSONAL_PATH.read_text())
    except Exception:
        pass

# --- Hybrid Retrieval ---
from core.retriever import hybrid_retrieve, RetrievalResult
from core.query_planner import plan_query, QueryType
from core.quality import QualityTracker, RetrievalMetric, compute_index_stats

# --- LLM ---
LLAMA_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080")

# --- Quality tracker ---
_quality = QualityTracker(DB_PATH)

# --- Cached status (refreshed periodically) ---
_status_cache = {"data": None, "last_updated": 0}
_STATUS_CACHE_TTL = 30  # seconds

# --- Connection pool ---
import threading
_thread_local = threading.local()


def _get_db_connection():
    """Get or create a thread-local database connection."""
    if not hasattr(_thread_local, 'db_conn') or _thread_local.db_conn is None:
        _thread_local.db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _thread_local.db_conn.row_factory = None
        _thread_local.db_conn.execute("PRAGMA journal_mode=WAL")
        _thread_local.db_conn.execute("PRAGMA cache_size=-64000")
        _thread_local.db_conn.execute("PRAGMA temp_store=MEMORY")
    return _thread_local.db_conn


def _llama_health():
    try:
        r = urllib.request.urlopen(LLAMA_URL + "/health", timeout=1)
        return r.status == 200
    except Exception:
        return False

def _llama_model():
    try:
        r = urllib.request.urlopen(LLAMA_URL + "/v1/models", timeout=1)
        d = json.loads(r.read().decode())
        models = d.get("models", d.get("data", []))
        if models:
            m = models[0]
            return m.get("id") or m.get("name") or "unknown"
    except Exception:
        pass
    return None

def llm_is_ready():
    return _llama_health()

def get_model_name():
    m = _llama_model()
    if m:
        return m
    if _llama_health():
        return "loaded"
    return "none"


def db_counts():
    if not DB_PATH.exists():
        return 0, 0
    conn = sqlite3.connect(DB_PATH)
    try:
        rec = conn.execute("SELECT COUNT(*) FROM content_records").fetchone()[0]
        chk = conn.execute("SELECT COUNT(*) FROM content_chunks").fetchone()[0]
        return rec, chk
    finally:
        conn.close()


def _build_personal_info():
    """Build personal profile info, skipping placeholder values."""
    if not _personal:
        return ""

    placeholders = {"your name", "your location", "your timezone", "your gpu", "your ram",
                    "your sbc", "your project", "your project 1", "your project 2",
                    "your ram_system", "your hardware"}

    lines = []
    name = _personal.get("name", "")
    location = _personal.get("location", "")
    timezone = _personal.get("timezone", "")
    hardware = _personal.get("hardware", {})
    tools = _personal.get("tools_used", [])
    projects = _personal.get("projects", [])
    interests = _personal.get("interests", [])

    def is_placeholder(val):
        if not val:
            return True
        return val.lower().strip() in placeholders

    if not is_placeholder(name):
        lines.append(f"- Name: {name}")
    if not is_placeholder(location):
        lines.append(f"- Location: {location}")
    if timezone and not is_placeholder(timezone):
        lines.append(f"- Timezone: {timezone}")
    if interests and not all(is_placeholder(i) for i in interests):
        real_interests = [i for i in interests if not is_placeholder(i)]
        if real_interests:
            lines.append(f"- Interests: {', '.join(real_interests)}")
    if hardware and not all(is_placeholder(v) for v in hardware.values()):
        parts = []
        for k, v in hardware.items():
            if not is_placeholder(v):
                label = k.replace("_", " ").title()
                parts.append(f"{label}: {v}")
        if parts:
            lines.append(f"- Hardware: {'; '.join(parts)}")
    if tools and not all(is_placeholder(t) for t in tools):
        real_tools = [t for t in tools if not is_placeholder(t)]
        if real_tools:
            lines.append(f"- AI Tools Used: {', '.join(real_tools)}")
    if projects and not all(is_placeholder(p) for p in projects):
        real_projects = [p for p in projects if not is_placeholder(p)]
        if real_projects:
            lines.append(f"- Projects: {', '.join(real_projects)}")

    return "\n".join(lines) if lines else ""


def generate_answer(question, context, query_plan=None):
    """Generate answer using LLM with retrieved context.

    Adapts the system prompt based on query type from the planner.
    """
    if not _llama_health():
        return "llama-server not running. Start: llama-server -m models/qwen3.5-4b.gguf --port 8080 -ngl 99"

    personal_info = _build_personal_info()

    name = _personal.get("name", "the user") if _personal else "the user"
    if name.lower().strip() in {"your name", ""}:
        name = "the user"

    # Adapt system prompt based on query type
    sys_prompt = (
        f"You are LocalBrain, {name}'s private local AI assistant.\n"
        f"You answer questions about {name}'s personal knowledge base.\n\n"
    )
    if personal_info:
        sys_prompt += f"ABOUT {name.upper()} (use this for personal questions):\n{personal_info}\n\n"

    # Type-specific instructions
    if query_plan and query_plan.query_type == QueryType.COMPARATIVE:
        sys_prompt += (
            "The user is asking a comparative question. You have multiple documents in context. "
            "Compare them carefully and identify differences, gaps, or missing elements. "
            "Reference specific passages when making comparisons.\n\n"
        )
    elif query_plan and query_plan.query_type == QueryType.ANALYTICAL:
        sys_prompt += (
            "The user wants an analytical answer. Synthesize information from multiple passages. "
            "Provide insights and patterns you observe across the retrieved context.\n\n"
        )

    sys_prompt += (
        "RULES:\n"
        "1. If the question asks about personal info (name, location, hardware, tools, projects, interests), "
        "answer from the ABOUT section above.\n"
        "2. For other questions, use the provided context passages as your primary source.\n"
        "3. Synthesize information from the context — if passages mention projects, topics, or activities, "
        "summarize what you find. Don't just say 'I don't have info' if there are relevant passages.\n"
        "4. If the context truly doesn't contain relevant information, say: "
        "'I don't have specific information about that in your knowledge base, but based on what I know...'\n"
        "5. Be concise and helpful. Cite passage numbers when possible.\n"
        "6. If you see a score or confidence indicator, prefer higher-scored passages."
    )

    if context:
        ctx_parts = []
        for i, c in enumerate(context[:8], 1):
            score_info = f" (relevance: {c.score:.1f})" if hasattr(c, 'score') and c.score else ""
            text = c.text[:500] if hasattr(c, 'text') else str(c.get("text", ""))[:500]
            source = c.source if hasattr(c, 'source') else c.get("source", "unknown")
            ctx_parts.append(f"[Passage {i}{score_info}, source: {source}]\n{text}")

        user_prompt = (
            f"Context from knowledge base:\n\n" + "\n\n".join(ctx_parts) +
            f"\n\nQuestion: {question}\n\n"
            f"Answer based on the ABOUT section and context above:"
        )
    else:
        user_prompt = (
            f"Question: {question}\n\n"
            f"Answer from the ABOUT section above. If the question is not about personal info, "
            f"say you don't have that information in the knowledge base."
        )

    payload = json.dumps({
        "model": "qwen3.5-9b.gguf",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 2048,
    }).encode("utf-8")

    req = urllib.request.Request(
        LLAMA_URL + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        resp = urllib.request.urlopen(req, timeout=120)
        body = json.loads(resp.read().decode("utf-8"))
        choices = body.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "No response generated.")
        return "Empty response."
    except urllib.error.HTTPError as e:
        return "LLM error: " + str(e.code)
    except Exception as e:
        return "LLM error: " + str(e)


def recent_activities(limit=10):
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            """SELECT cr.title, cr.category, cr.created_at
               FROM content_records cr
               ORDER BY cr.created_at DESC
               LIMIT ?""",
            (limit,)
        ).fetchall()
        items = []
        for r in rows:
            title = (r[0] or "Untitled")[:80]
            cat = r[1] or "unknown"
            created = str(r[2] or "")[:10]
            items.append({"title": title, "category": cat, "created_at": created})
        return items
    finally:
        conn.close()


def _format_timestamp(ts) -> str:
    if not ts:
        return "unknown"
    ts_str = str(ts)
    if len(ts_str) == 13 and ts_str.isdigit():
        from datetime import datetime
        try:
            return datetime.utcfromtimestamp(int(ts_str) / 1000).strftime("%Y-%m-%d")
        except Exception:
            pass
    if "T" in ts_str:
        return ts_str[:10]
    return ts_str[:10]


# --- HTTP Handler ---
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(RUNTIME_DIR), **kwargs)

    def log_message(self, fmt, *args):
        # Log to stderr for debugging
        sys.stderr.write(f"[HTTP] {fmt % args}\n")

    def _json(self, data, status=200):
        body = json.dumps(data, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/status":
            import time as _time
            now = _time.time()
            cached = _status_cache.get("data")
            if cached and (now - _status_cache["last_updated"]) < _STATUS_CACHE_TTL:
                self._json(cached)
                return

            rec, chk = db_counts()
            # Skip expensive index_stats on every request
            self._json({
                "records": rec,
                "chunks": chk,
                "retrieval": "hybrid_v2",
                "model": get_model_name(),
                "llm_ready": llm_is_ready(),
            })
            _status_cache["data"] = {
                "records": rec,
                "chunks": chk,
                "retrieval": "hybrid_v2",
                "model": get_model_name(),
                "llm_ready": llm_is_ready(),
            }
            _status_cache["last_updated"] = now
        elif self.path == "/api/recent":
            self._json({"items": recent_activities(10)})
        elif self.path == "/api/quality":
            report = _quality.report(days=7)
            self._json(report.to_dict())
        elif self.path == "/api/index-stats":
            stats = compute_index_stats(DB_PATH)
            self._json(stats)
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/chat":
            self._handle_chat()
        else:
            self._json({"error": "Not found"}, 404)

    def _handle_chat(self):
        import traceback as _tb
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode()) if length else {}
        except Exception as e:
            self._json({"error": "Bad request: " + str(e)}, 400)
            return

        question = payload.get("question", "").strip()
        do_generate = payload.get("generate_answer", False)
        top_k = int(payload.get("top_k", 10))
        force_strategy = payload.get("strategy", None)

        if not question:
            self._json({"error": "Question required"}, 400)
            return

        try:
            t0 = time.perf_counter()

            # Step 1: Plan the query
            query_plan = plan_query(question, top_k)
            if force_strategy:
                query_plan.strategy = force_strategy

            # Step 2: Retrieve using planned strategy
            results, retrieval_stats = hybrid_retrieve(
                question,
                top_k=query_plan.top_k,
                db_path=DB_PATH,
                use_reranker=query_plan.use_reranker,
                use_vectors=query_plan.use_vectors,
                strategy=query_plan.strategy,
                fast_mode=(force_strategy not in ("hybrid", "vector_only", "long_context")),
            )

            # Step 3: Record quality metrics
            metric = RetrievalMetric(
                query=question,
                query_type=query_plan.query_type.value,
                strategy=retrieval_stats.strategy,
                bm25_candidates=retrieval_stats.bm25_candidates,
                vector_candidates=retrieval_stats.vector_candidates,
                final_count=len(results),
                elapsed_ms=retrieval_stats.elapsed_ms,
                top_score=results[0].score if results else 0.0,
                avg_score=sum(r.score for r in results) / max(len(results), 1),
                score_spread=(results[0].score - results[-1].score) if len(results) > 1 else 0.0,
                sources_count=len(set(r.source for r in results)),
            )
            _quality.record(metric)

            # Step 4: Generate answer
            if do_generate and (results or _personal):
                answer = generate_answer(question, results, query_plan)
            elif results:
                answer = f"Found {len(results)} relevant passages."
            else:
                answer = "No relevant passages found in your knowledge base."

            elapsed = round((time.perf_counter() - t0) * 1000, 1)

            # Step 5: Build response with full provenance
            self._json({
                "question": question,
                "answer": answer,
                "passages": [r.to_dict() for r in results],
                "elapsed_ms": elapsed,
                "query_plan": query_plan.to_dict(),
                "retrieval_stats": retrieval_stats.to_dict(),
            })

        except Exception as e:
            _tb.print_exc()
            try:
                self._json({"error": f"Internal error: {str(e)}"}, 500)
            except:
                pass


class Server(ThreadingMixIn, HTTPServer):
    """Threaded HTTP server."""
    allow_reuse_address = True
    daemon_threads = True


def main():
    for d in [DATA_DIR, RUNTIME_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    rec, chk = db_counts()

    print("=" * 50)
    print("LocalBrain - Personal AI Node v2")
    print("=" * 50)
    print(f"  Web UI:    http://localhost:{PORT}")
    print(f"  Database:  {DB_PATH}")
    print(f"  Records:   {rec}")
    print(f"  Chunks:    {chk}")
    print(f"  Retrieval: Hybrid v2 (BM25 + Vector + Reranker + Query Planning)")
    print(f"  LLM:       {'Ready' if llm_is_ready() else 'Not loaded'}")
    print(f"  Model:     {get_model_name()}")
    print(f"  Profile:   {'Yes' if _personal else 'No'}")
    print("=" * 50)
    print("Press Ctrl+C to stop")

    httpd = Server((HOST, PORT), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        httpd.server_close()


if __name__ == "__main__":
    main()
