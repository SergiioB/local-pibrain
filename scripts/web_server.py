#!/usr/bin/env python3
"""
LocalBrain - Web server with hybrid RAG + LLM.
BM25 keyword retrieval + cross-encoder reranking. No vector DB needed.
"""
import json
import os
import sqlite3
import sys
import time
import urllib.request
from http.server import HTTPServer, SimpleHTTPRequestHandler
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

# --- Hybrid Search (BM25 + cross-encoder reranking) ---
try:
    from core.hybrid_search import search as hybrid_search
    HAS_HYBRID = True
except ImportError:
    HAS_HYBRID = False

# --- BM25 fallback ---
try:
    from core.bm25 import retrieve_chunks as bm25_retrieve
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

# --- LLM ---
LLAMA_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080")

def _llama_health():
    try:
        r = urllib.request.urlopen(LLAMA_URL + "/health", timeout=3)
        return r.status == 200
    except Exception:
        return False

def _llama_model():
    try:
        r = urllib.request.urlopen(LLAMA_URL + "/v1/models", timeout=3)
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


def retrieve(query, top_k):
    """Retrieve using hybrid search (BM25 + cross-encoder)."""
    if HAS_HYBRID:
        return hybrid_search(query, top_k, DB_PATH)
    if HAS_BM25:
        return bm25_retrieve(query, top_k)
    # Ultimate fallback
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(DB_PATH)
    results = []
    try:
        rows = conn.execute(
            """SELECT cc.id, cc.chunk_text, cr.title, sf.source_type, cr.created_at
               FROM content_chunks cc
               JOIN content_records cr ON cc.content_record_id = cr.id
               JOIN source_files sf ON cr.source_file_id = sf.id
               WHERE cc.chunk_text LIKE ?
               LIMIT ?""",
            ("%" + query + "%", top_k)
        ).fetchall()
        for r in rows:
            results.append({
                "id": r[0], "text": r[1][:500], "title": r[2] or "Untitled",
                "source": r[3], "created_at": r[4] or "unknown", "score": 0.5,
            })
    finally:
        conn.close()
    return results


def _build_personal_info():
    """Build personal profile info, skipping placeholder values."""
    if not _personal:
        return ""

    # Detect placeholder values — skip fields with template text
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

    # Only include info if real values exist (not placeholders)
    has_real_name = not is_placeholder(name)
    has_real_location = not is_placeholder(location)

    if has_real_name:
        lines.append(f"- Name: {name}")
    if has_real_location:
        lines.append(f"- Location: {location}")
    if timezone and not is_placeholder(timezone):
        lines.append(f"- Timezone: {timezone}")
    if interests and not all(is_placeholder(i) for i in interests):
        real_interests = [i for i in interests if not is_placeholder(i)]
        if real_interests:
            lines.append(f"- Interests: {', '.join(real_interests)}")
    if hardware and not all(is_placeholder(v) for v in hardware.values()):
        parts = [f"{k}: {v}" for k, v in hardware.items() if not is_placeholder(v)]
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

    if not lines:
        return ""

    return "\n".join(lines)


def generate_answer(question, context):
    if not _llama_health():
        return "llama-server not running. Start: llama-server -m models/qwen3.5-4b.gguf --port 8080 -ngl 99"

    personal_info = _build_personal_info()

    # Get display name — skip if it's a placeholder
    name = _personal.get("name", "the user") if _personal else "the user"
    if name.lower().strip() in {"your name", ""}:
        name = "the user"

    sys_prompt = (
        f"You are LocalBrain, {name}'s private local AI assistant.\n"
        f"You answer questions about {name}'s personal knowledge base.\n\n"
    )
    if personal_info:
        sys_prompt += f"ABOUT {name.upper()} (use this for personal questions):\n{personal_info}\n\n"
    sys_prompt += (
        "RULES:\n"
        "1. If the question asks about personal info (name, location, hardware, tools, projects, interests), "
        "answer from the ABOUT section above.\n"
        "2. For other questions, answer ONLY from the provided context passages.\n"
        "3. If context doesn't contain the answer, say: 'I don't have that information in your knowledge base.'\n"
        "4. Be concise. Cite sources when possible."
    )

    if context:
        ctx = "\n\n".join([c["text"][:500] for c in context[:5]])
        user_prompt = (
            f"Context from knowledge base:\n{ctx}\n\n"
            f"Question: {question}\n\n"
            f"Answer based on the ABOUT section and context above:"
        )
    else:
        user_prompt = (
            f"Question: {question}\n\n"
            f"Answer from the ABOUT section above. If the question is not about personal info, "
            f"say you don't have that information in the knowledge base."
        )

    payload = json.dumps({
        "model": "qwen3.5-4b.gguf",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
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


# --- HTTP Handler ---
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(RUNTIME_DIR), **kwargs)

    def log_message(self, fmt, *args):
        pass

    def _json(self, data, status=200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/status":
            rec, chk = db_counts()
            self._json({
                "records": rec,
                "chunks": chk,
                "retrieval": "hybrid" if HAS_HYBRID else ("bm25" if HAS_BM25 else "keyword"),
                "model": get_model_name(),
                "llm_ready": llm_is_ready(),
            })
        elif self.path == "/api/recent":
            self._json({"items": recent_activities(10)})
        else:
            super().do_GET()

    def do_POST(self):
        if self.path != "/api/chat":
            self._json({"error": "Not found"}, 404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode()) if length else {}
        except Exception as e:
            self._json({"error": "Bad request: " + str(e)}, 400)
            return

        question = payload.get("question", "").strip()
        do_generate = payload.get("generate_answer", False)
        top_k = int(payload.get("top_k", 5))

        if not question:
            self._json({"error": "Question required"}, 400)
            return

        t0 = time.perf_counter()
        passages = retrieve(question, top_k)

        if do_generate and (passages or _personal):
            answer = generate_answer(question, passages)
        elif passages:
            answer = "Found %d relevant passages." % len(passages)
        else:
            answer = "No relevant passages found in your knowledge base."

        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        self._json({
            "question": question,
            "answer": answer,
            "passages": passages,
            "elapsed_ms": elapsed,
        })


class Server(HTTPServer):
    allow_reuse_address = True


def main():
    for d in [DATA_DIR, RUNTIME_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    rec, chk = db_counts()
    print("=" * 50)
    print("LocalBrain - Personal AI Node")
    print("=" * 50)
    print(f"  Web UI:    http://localhost:{PORT}")
    print(f"  Database:  {DB_PATH}")
    print(f"  Records:   {rec}")
    print(f"  Chunks:    {chk}")
    print(f"  Retrieval: {'Hybrid (BM25 + reranker)' if HAS_HYBRID else 'BM25'}")
    print(f"  LLM:       {'Ready' if llm_is_ready() else 'Not loaded'}")
    print(f"  Model:     {get_model_name()}")
    print(f"  Profile:   {'Yes' if _personal else 'No (runtime/personal.json missing)'}")
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
