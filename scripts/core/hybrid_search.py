"""
Hybrid retrieval: BM25 (FTS5-based) + optional cross-encoder reranking.
Optimized for speed: FTS5 does the heavy lifting, cross-encoder is optional.
Adds LRU cache for repeated queries.
"""
import os
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path
from typing import List, Dict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Prevent thread spam

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "state.db"

# Retrieval parameters
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BM25_CANDIDATES = 50     # FTS5 is fast, fewer candidates needed
RELEVANCE_THRESHOLD = 0.0
MAX_PASSAGES = 10

_reranker = None


# --- Simple LRU cache for query results ---
class LRUCache:
    def __init__(self, maxlen=256):
        self._cache = OrderedDict()
        self._maxlen = maxlen

    def get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._maxlen:
            self._cache.popitem(last=False)


_result_cache = LRUCache(maxlen=256)


def _load_reranker():
    """Load cross-encoder reranker model (lazy, only if explicitly wanted)."""
    global _reranker
    if _reranker is not None:
        return _reranker

    try:
        import warnings
        warnings.filterwarnings('ignore')
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL)
        print(f"  Reranker loaded: {RERANKER_MODEL}")
        return _reranker
    except Exception as e:
        print(f"  Reranker unavailable: {e}")
        _reranker = "none"
        return _reranker


def bm25_search(db_path: Path, query: str, top_k: int = BM25_CANDIDATES) -> List[dict]:
    """Retrieve candidates using FTS5-powered BM25 search."""
    try:
        from core.bm25 import retrieve_chunks
        results = retrieve_chunks(query, top_k, db_path)
        if results:
            return results
    except ImportError:
        pass

    # Fallback: direct FTS5 query
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    results = []
    try:
        terms = [t for t in query.lower().split() if len(t) > 2]
        if not terms:
            return []

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
            text_lower = (text or "").lower()
            for t in terms:
                score += min(text_lower.count(t) * 0.5, 2.0)
            score += len([t for t in terms if t in text_lower]) * 1.0

            results.append({
                "id": chunk_id,
                "text": (text or "")[:500],
                "title": title or "Untitled",
                "source": source or "unknown",
                "created_at": str(created_at) if created_at else "unknown",
                "score": round(score, 3),
            })
    except sqlite3.OperationalError:
        pass
    finally:
        conn.close()

    return results


def deduplicate_results(results: List[dict], threshold: float = 0.8) -> List[dict]:
    """Remove near-duplicate passages based on text overlap."""
    unique = []
    seen_texts = []

    for r in results:
        text = r.get("text", "")[:200]
        is_dup = False
        for seen in seen_texts:
            words_a = set(text.lower().split())
            words_b = set(seen.lower().split())
            if not words_a or not words_b:
                continue
            overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
            if overlap > threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(r)
            seen_texts.append(text)

    return unique


def rerank_with_cross_encoder(query: str, candidates: List[dict], top_k: int = 10) -> List[dict]:
    """Rerank candidates using cross-encoder model (optional, may be slow on CPU)."""
    reranker = _load_reranker()

    if reranker == "none" or not candidates:
        return candidates[:top_k]

    pairs = [(query, c["text"]) for c in candidates]

    try:
        scores = reranker.predict(pairs, show_progress_bar=False)
    except Exception:
        return candidates[:top_k]

    scored = []
    for i, score in enumerate(scores):
        score = float(score)
        if score > RELEVANCE_THRESHOLD:
            item = candidates[i].copy()
            item["score"] = round(score, 4)
            scored.append(item)

    scored.sort(key=lambda x: x["score"], reverse=True)
    scored = deduplicate_results(scored)

    return scored[:top_k]


def search(query: str, top_k: int = 10, db_path: Path = DB_PATH,
           use_reranker: bool = False) -> List[dict]:
    """
    Hybrid search: FTS5 BM25 retrieval + optional cross-encoder reranking.

    1. FTS5 retrieves top-K candidates (fast keyword match)
    2. Optionally cross-encoder reranks (semantic understanding, slower on CPU)
    3. Deduplicate: remove near-duplicate passages
    4. Return top-K
    """
    # Check cache first
    cache_key = f"{query}|{top_k}|{use_reranker}"
    cached = _result_cache.get(cache_key)
    if cached is not None:
        return cached

    # Step 1: BM25 via FTS5 (fast)
    candidates = bm25_search(db_path, query, BM25_CANDIDATES)

    if not candidates:
        _result_cache.put(cache_key, [])
        return []

    # Step 2: Optional cross-encoder reranking
    if use_reranker:
        results = rerank_with_cross_encoder(query, candidates, min(top_k, MAX_PASSAGES))
    else:
        # Just deduplicate BM25 results
        results = deduplicate_results(candidates)
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        results = results[:min(top_k, MAX_PASSAGES)]

    _result_cache.put(cache_key, results)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hybrid retrieval (FTS5 BM25 + optional cross-encoder)")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--rerank", action="store_true", help="Use cross-encoder reranking")
    args = parser.parse_args()

    if args.query:
        t0 = time.perf_counter()
        results = search(args.query, args.top_k, args.db, use_reranker=args.rerank)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"\nFound {len(results)} results for '{args.query}' ({elapsed:.0f}ms):\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r['source']}] score={r['score']:.2f} {r['title'][:60]}...")
            print(f"   {r['text'][:150]}...")
            print()
    else:
        print("Usage: python hybrid_search.py 'your query here'")
