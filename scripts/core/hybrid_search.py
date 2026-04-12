"""
Hybrid retrieval: BM25 + cross-encoder reranking.
Optimized for quality: filters irrelevant results, deduplicates, query expansion.
"""
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.modules['tensorboard'] = None
sys.modules['torch.utils.tensorboard'] = type(sys)('fake')

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "state.db"

# Retrieval parameters
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BM25_CANDIDATES = 100    # More candidates = better recall
RELEVANCE_THRESHOLD = 0.0  # Cross-encoder score threshold (0 = neutral)
MAX_PASSAGES = 10         # Hard limit for returned passages

_reranker = None


def _load_reranker():
    """Load cross-encoder reranker model."""
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


def expand_query(query: str) -> str:
    """Expand query with related terms for better BM25 recall."""
    expansions = {
        "work": "job company employer career position",
        "live": "location city country residence address",
        "hardware": "gpu cpu ram computer device machine",
        "model": "llm ai model weights checkpoint gguf",
        "project": "code repository github development build",
        "fan": "cooling thermal temperature pwm",
        "code": "programming python javascript typescript",
    }
    expanded = query.lower()
    for keyword, terms in expansions.items():
        if keyword in expanded:
            return query + " " + terms
    return query


def bm25_search(db_path: Path, query: str, top_k: int = BM25_CANDIDATES) -> List[dict]:
    """Retrieve candidates using BM25 keyword search."""
    try:
        from core.bm25 import retrieve_chunks
        results = retrieve_chunks(query, top_k, db_path)
        if results:
            return results
    except ImportError:
        pass

    # Fallback: keyword search with query expansion
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    results = []
    try:
        # Expand query for better recall
        expanded = expand_query(query)
        stop_words = {'the', 'and', 'for', 'with', 'this', 'that', 'what', 'how', 'why', 'where', 'when', 'do', 'i', 'me', 'my'}
        terms = [t for t in expanded.lower().split() if len(t) > 2 and t not in stop_words]
        if not terms:
            terms = query.lower().split()[:5]

        conditions = " OR ".join(["cc.chunk_text LIKE ?" for _ in terms])
        params = [f"%{t}%" for t in terms]

        rows = conn.execute(
            f"""SELECT cc.id, cc.chunk_text, cr.title, sf.source_type, cr.created_at
               FROM content_chunks cc
               JOIN content_records cr ON cc.content_record_id = cr.id
               JOIN source_files sf ON cr.source_file_id = sf.id
               WHERE {conditions}
               LIMIT ?""",
            params + [top_k]
        ).fetchall()

        for row in rows:
            results.append({
                "id": row[0],
                "text": row[1][:500],
                "title": row[2] or "Untitled",
                "source": row[3],
                "created_at": str(row[4]) if row[4] else "unknown",
                "score": 0.5,
            })
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
            # Simple Jaccard similarity on words
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
    """Rerank candidates using cross-encoder model."""
    reranker = _load_reranker()

    if reranker == "none" or not candidates:
        return candidates[:top_k]

    # Create query-document pairs
    pairs = [(query, c["text"]) for c in candidates]

    # Get similarity scores
    scores = reranker.predict(pairs, show_progress_bar=False)

    # Attach scores and filter by threshold
    scored = []
    for i, score in enumerate(scores):
        score = float(score)
        # Only keep relevant results (positive cross-encoder score)
        if score > RELEVANCE_THRESHOLD:
            item = candidates[i].copy()
            item["score"] = round(score, 4)
            scored.append(item)

    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Deduplicate
    scored = deduplicate_results(scored)

    return scored[:top_k]


def search(query: str, top_k: int = 10, db_path: Path = DB_PATH) -> List[dict]:
    """
    Hybrid search: BM25 retrieval + cross-encoder reranking.

    1. BM25 retrieves top-100 candidates (fast keyword match)
    2. Cross-encoder reranks top-100 (semantic understanding)
    3. Filter: only keep results with positive relevance score
    4. Deduplicate: remove near-duplicate passages
    5. Return top-K
    """
    # Step 1: BM25 retrieval (expanded query for better recall)
    candidates = bm25_search(db_path, query, BM25_CANDIDATES)

    if not candidates:
        return []

    # Step 2: Cross-encoder reranking with filtering
    results = rerank_with_cross_encoder(query, candidates, min(top_k, MAX_PASSAGES))

    return results


if __name__ == "__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser(description="Hybrid retrieval (BM25 + cross-encoder)")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()

    if args.query:
        t0 = time.perf_counter()
        results = search(args.query, args.top_k, args.db)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"\nFound {len(results)} results for '{args.query}' ({elapsed:.0f}ms):\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r['source']}] score={r['score']:.2f} {r['title'][:60]}...")
            print(f"   {r['text'][:150]}...")
            print()
    else:
        print("Usage: python hybrid_search.py 'your query here'")
