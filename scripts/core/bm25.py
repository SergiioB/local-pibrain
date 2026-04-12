#!/usr/bin/env python3
"""
BM25 retriever for RAG - uses SQLite FTS5 for fast keyword search.
No longer loads all chunks into memory; queries FTS5 directly.
Falls back to in-memory BM25 only if FTS5 is unavailable.
"""
import sys
import sqlite3
import math
import re
from pathlib import Path
from threading import Lock

sys.stdout.reconfigure(encoding='utf-8')

DB_PATH = Path("data/state.db")

STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'you', 'your', 'he', 'him', 'his',
    'she', 'her', 'it', 'its', 'they', 'them', 'their', 'what', 'which', 'who', 'whom',
    'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
    'by', 'for', 'with', 'about', 'against', 'between', 'through', 'during', 'before',
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
    'el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'en', 'y', 'o', 'que', 'por',
    'para', 'con', 'sin', 'se', 'lo', 'mi', 'tu', 'su', 'es', 'son', 'tiene', 'han',
}

def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r'[a-z0-9\u00c0-\u024f]{3,}', text)
    return [t for t in tokens if t not in STOP_WORDS]


def _fts5_available(db_path: Path) -> bool:
    """Check if FTS5 table exists and is populated."""
    if not db_path.exists():
        return False
    conn = sqlite3.connect(db_path)
    try:
        cnt = conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
        return cnt > 0
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return False
    finally:
        conn.close()


def _search_fts5(db_path: Path, query: str, top_k: int) -> list[dict]:
    """Fast FTS5 keyword search on chunk_text."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    results = []
    try:
        terms = tokenize(query)
        if not terms:
            terms = query.lower().split()[:10]

        fts_query = " OR ".join(terms)

        rows = conn.execute(
            """SELECT rowid, chunk_text, title, source_type, content_record_id
               FROM chunks_fts
               WHERE chunks_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (fts_query, top_k),
        ).fetchall()

        # Batch fetch created_at
        record_ids = [r[4] for r in rows if r[4]]
        created_at_map = {}
        if record_ids:
            placeholders = ",".join("?" * len(record_ids))
            for row2 in conn.execute(
                f"SELECT id, created_at FROM content_records WHERE id IN ({placeholders})",
                record_ids
            ):
                created_at_map[row2[0]] = row2[1]

        for row in rows:
            chunk_id, text, title, source, record_id = row
            created_at = created_at_map.get(record_id)
            score = 0.0
            text_lower = text.lower() if text else ""
            for t in terms:
                count = text_lower.count(t)
                score += min(count * 0.5, 2.0)
            score += len([t for t in terms if t in text_lower]) * 1.0

            results.append({
                'id': chunk_id,
                'text': (text or "")[:500],
                'title': title or 'Untitled',
                'source': source or 'unknown',
                'created_at': str(created_at) if created_at else 'unknown',
                'score': round(score, 3),
            })
    except sqlite3.OperationalError:
        pass
    finally:
        conn.close()

    return results


class BM25Retriever:
    """BM25 retriever with in-memory inverted index (fallback only)."""

    def __init__(self, db_path: Path = DB_PATH):
        self.k1 = 1.5
        self.b = 0.75
        self.avg_doc_len = 0
        self.N = 0
        self.idf = {}
        self.documents = []
        self.db_path = db_path

    def _build_index(self):
        """Build in-memory index only as fallback when FTS5 is unavailable."""
        if not self.db_path.exists():
            return

        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                """SELECT cc.id, cc.chunk_text, cr.title, sf.source_type, cr.created_at
                   FROM content_chunks cc
                   JOIN content_records cr ON cc.content_record_id = cr.id
                   JOIN source_files sf ON cr.source_file_id = sf.id
                   LIMIT 50000"""
            ).fetchall()
        finally:
            conn.close()

        for row in rows:
            chunk_id, text, title, source, created_at = row
            if not text or len(text) < 20:
                continue
            tokens = tokenize(text)
            if not tokens:
                continue
            self.documents.append({
                'id': chunk_id,
                'text': text[:500],
                'title': title or 'Untitled',
                'source': source,
                'created_at': created_at or 'unknown',
                'tokens': tokens,
            })

        self.N = len(self.documents)
        if self.N == 0:
            return

        term_doc_freq = {}
        total_tokens = 0
        for doc in self.documents:
            total_tokens += len(doc['tokens'])
            seen = set()
            for t in doc['tokens']:
                if t not in seen:
                    term_doc_freq[t] = term_doc_freq.get(t, 0) + 1
                    seen.add(t)

        self.avg_doc_len = total_tokens / self.N

        for term, df in term_doc_freq.items():
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.N == 0:
            return []

        expanded = expand_query(query)
        query_tokens = tokenize(expanded)
        if not query_tokens:
            return []

        orig_tokens = tokenize(query)
        scores = []
        for doc in self.documents:
            score = 0.0
            doc_len = len(doc['tokens'])
            doc_term_freq = {}
            for t in doc['tokens']:
                doc_term_freq[t] = doc_term_freq.get(t, 0) + 1

            for qt in query_tokens:
                if qt not in doc_term_freq:
                    continue
                tf = doc_term_freq[qt]
                idf = self.idf.get(qt, 0.0)
                if idf <= 0:
                    continue
                num = tf * (self.k1 + 1)
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                score += idf * num / denom

            for qt in orig_tokens:
                if qt in doc_term_freq:
                    score += 2.0

            if score > 0:
                scores.append((score, doc))

        scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in scores[:top_k]:
            results.append({
                'id': doc['id'],
                'text': doc['text'],
                'title': doc['title'],
                'source': doc['source'],
                'created_at': doc['created_at'],
                'score': round(score, 3),
            })

        return results


# Global cached instance
_retriever = None
_retriever_lock = Lock()

def get_retriever(db_path: Path = DB_PATH):
    """Get or create the cached BM25 retriever."""
    global _retriever
    with _retriever_lock:
        if _retriever is None:
            _retriever = BM25Retriever(db_path)
            # Only build in-memory index if FTS5 is not available
            if not _fts5_available(db_path):
                _retriever._build_index()
        return _retriever


def retrieve_chunks(query: str, limit: int = 5, db_path: Path = DB_PATH) -> list[dict]:
    """Public API - retrieve chunks using FTS5 (preferred) or in-memory BM25."""
    # Always try FTS5 first - it's fast and doesn't load everything into memory
    if _fts5_available(db_path):
        results = _search_fts5(db_path, query, limit)
        if results:
            return results

    # Fallback to in-memory BM25
    retriever = get_retriever(db_path)
    return retriever.search(query, limit)


# Query expansion patterns
QUERY_EXPANSIONS = {
    "age": ["birth", "born", "year", "old"],
    "project": ["repo", "github", "code", "app", "bot", "service"],
    "work": ["task", "build", "implement", "fix", "feature", "code"],
    "fan": ["pwm", "thermal", "cooling", "temperature", "heat"],
    "model": ["llm", "ai", "qwen", "gpt", "inference", "gguf"],
}

def expand_query(query: str) -> str:
    """Add related terms for better BM25 recall."""
    q_lower = query.lower()
    expanded = query
    for keyword, related_terms in QUERY_EXPANSIONS.items():
        if keyword in q_lower:
            expanded += " " + " ".join(related_terms[:3])
            break
    return expanded
