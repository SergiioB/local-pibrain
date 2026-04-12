"""
Hybrid retrieval: BM25 + cross-encoder reranking.
No vector database needed. Fast + accurate.

Strategy:
1. BM25 retrieves top-N candidates (fast keyword match)
2. Cross-encoder reranks candidates (semantic understanding)
3. Return top-K after reranking

This is superior to dense embeddings because:
- Cross-encoders see query+text together (not independent embeddings)
- No embedding generation or storage needed
- BM25 handles exact matches, cross-encoder handles semantics
"""
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.modules['tensorboard'] = None
sys.modules['torch.utils.tensorboard'] = type(sys)('fake')

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "state.db"

# Cross-encoder model - small but excellent for reranking
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BM25_CANDIDATES = 50  # How many BM25 results to rerank

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
        print("  Falling back to BM25-only retrieval")
        _reranker = "none"
        return _reranker


def bm25_search(db_path: Path, query: str, top_k: int = BM25_CANDIDATES) -> List[dict]:
    """Retrieve candidates using BM25 keyword search."""
    try:
        from core.bm25 import retrieve_chunks
        results = retrieve_chunks(query, top_k, db_path)
        return results
    except ImportError:
        pass
    
    # Fallback: simple LIKE search
    if not db_path.exists():
        return []
    
    conn = sqlite3.connect(db_path)
    results = []
    try:
        # Expand query terms
        stop_words = {'the', 'and', 'for', 'with', 'this', 'that', 'what', 'how', 'why', 'where', 'when'}
        terms = [t.lower() for t in query.split() if len(t) > 2 and t.lower() not in stop_words]
        if not terms:
            terms = query.split()[:3]
        
        conditions = " OR ".join(["cc.chunk_text LIKE ?" for _ in terms])
        params = [f"%{t}%" for t in terms]
        
        rows = conn.execute(
            f"""SELECT cc.id, cc.chunk_text, cr.title, sf.source_type, cr.created_at
               FROM content_chunks cc
               JOIN content_records cr ON cc.content_record_id = cr.id
               JOIN source_files sf ON cr.source_file_id = sf.id
               WHERE {conditions}
               ORDER BY cc.id DESC
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


def rerank_with_cross_encoder(query: str, candidates: List[dict], top_k: int = 10) -> List[dict]:
    """Rerank candidates using cross-encoder model."""
    reranker = _load_reranker()
    
    if reranker == "none" or not candidates:
        return candidates[:top_k]
    
    # Create query-document pairs for cross-encoder
    pairs = [(query, c["text"]) for c in candidates]
    
    # Get similarity scores
    scores = reranker.predict(pairs, show_progress_bar=False)
    
    # Attach scores to candidates
    for i, score in enumerate(scores):
        candidates[i]["score"] = round(float(score), 4)
    
    # Sort by score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    return candidates[:top_k]


def search(query: str, top_k: int = 10, db_path: Path = DB_PATH) -> List[dict]:
    """
    Hybrid search: BM25 retrieval + cross-encoder reranking.
    
    This is superior to vector embeddings because:
    1. No embedding generation or storage needed
    2. Cross-encoder sees query+document together (not independent)
    3. BM25 handles exact keyword matches perfectly
    4. Cross-encoder handles semantic understanding
    """
    # Step 1: BM25 retrieval
    candidates = bm25_search(db_path, query, BM25_CANDIDATES)
    
    if not candidates:
        return []
    
    # Step 2: Cross-encoder reranking (if available)
    results = rerank_with_cross_encoder(query, candidates, top_k)
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hybrid retrieval (BM25 + cross-encoder)")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()
    
    if args.query:
        results = search(args.query, args.top_k, args.db)
        print(f"\nFound {len(results)} results for '{args.query}':\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r['source']}] {r['title'][:60]}... (score: {r['score']})")
            print(f"   {r['text'][:150]}...")
            print()
    else:
        print("Usage: python hybrid_search.py 'your query here'")
