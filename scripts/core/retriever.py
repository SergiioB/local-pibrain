#!/usr/bin/env python3
"""
Unified hybrid retriever for LocalBrain.

Combines BM25 (FTS5), vector similarity, and optional cross-encoder reranking.
Inspired by the "RAG vs Long Context" analysis:
  - For infinite datasets: retrieval layer filters to what fits (RAG approach)
  - For bounded queries: can use long context when appropriate
  - Cross-encoder reranking solves the "needle in haystack" problem
  - Deduplication and quality scoring reduce silent failures

Architecture:
  1. BM25 (FTS5) → fast keyword candidates (broad recall)
  2. Vector search (sqlite-vec) → semantic similarity (semantic recall)
  3. Merge + deduplicate → combine both signals
  4. Cross-encoder rerank → precision boost (optional, slower)
  5. Recency + importance weighting → final ranking
"""

import math
import re
import sqlite3
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "state.db"


# ─── Data classes ───────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """A single retrieved passage with full provenance."""
    chunk_id: int
    record_id: int
    text: str
    title: str
    source: str
    category: str
    created_at: str
    score: float = 0.0
    bm25_score: float = 0.0
    vector_score: float = 0.0
    reranker_score: float = 0.0
    matched_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.chunk_id,
            "record_id": self.record_id,
            "text": self.text[:500],
            "title": self.title,
            "source": self.source,
            "category": self.category,
            "created_at": self.created_at,
            "score": round(self.score, 4),
            "bm25_score": round(self.bm25_score, 4),
            "vector_score": round(self.vector_score, 4),
            "reranker_score": round(self.reranker_score, 4),
            "matched_terms": self.matched_terms,
        }


@dataclass
class RetrievalStats:
    """Statistics about a retrieval operation."""
    query: str
    bm25_candidates: int = 0
    vector_candidates: int = 0
    merged_candidates: int = 0
    reranked: bool = False
    final_count: int = 0
    elapsed_ms: float = 0.0
    strategy: str = "hybrid"

    def to_dict(self) -> dict:
        return self.__dict__


# ─── Stop words ─────────────────────────────────────────────────────────

STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'you', 'your', 'he', 'him',
    'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their', 'what', 'which',
    'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
    'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'en', 'y', 'o', 'que',
    'por', 'para', 'con', 'sin', 'se', 'lo', 'mi', 'tu', 'su', 'es', 'son',
}


def tokenize(text: str) -> List[str]:
    """Tokenize and remove stop words."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = text.encode("ascii", "ignore").decode("ascii")
    tokens = re.findall(r'[a-z0-9]{3,}', text)
    return [t for t in tokens if t not in STOP_WORDS]


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", "ignore").decode("ascii")
    return text.lower()


# ─── Query expansion ────────────────────────────────────────────────────

QUERY_EXPANSIONS = {
    "docker": ["container", "podman", "deploy"],
    "gpu": ["nvidia", "cuda", "vram"],
    "database": ["sqlite", "db", "postgres"],
    "server": ["api", "http", "backend"],
    "ai": ["llm", "model", "inference", "embedding"],
    "project": ["repo", "codebase", "app", "bot"],
    "fan": ["cooling", "thermal", "temperature", "pwm"],
    "android": ["mobile", "kotlin", "app"],
    "hardware": ["cpu", "ram", "gpu", "device"],
    "network": ["dns", "proxy", "ip"],
    "deploy": ["docker", "production", "release"],
    "model": ["llm", "gguf", "inference", "qwen"],
}


def expand_query(query: str) -> List[str]:
    """Return original + expanded query variants."""
    variants = [query]
    terms = tokenize(query)
    expanded = list(terms)
    for t in terms:
        if t in QUERY_EXPANSIONS:
            expanded.extend(QUERY_EXPANSIONS[t][:2])
    if expanded != terms:
        variants.append(" ".join(expanded))
    return variants


# ─── Recency weighting ─────────────────────────────────────────────────

def recency_weight(created_at, decay_days: int = 365) -> float:
    """Exponential recency decay: 1.0 (today) → ~0.37 (1 year old)."""
    if not created_at:
        return 0.5
    from datetime import datetime
    ts_str = str(created_at)
    try:
        if len(ts_str) == 13 and ts_str.isdigit():
            dt = datetime.utcfromtimestamp(int(ts_str) / 1000)
        elif "T" in ts_str:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if dt.tzinfo:
                dt = dt.replace(tzinfo=None)
        else:
            dt = datetime.fromisoformat(ts_str[:10])
        days_old = max(0, (datetime.now() - dt).days)
        return math.exp(-days_old / decay_days)
    except Exception:
        return 0.5


# ─── Stage 1: BM25 via FTS5 ────────────────────────────────────────────

def bm25_search(conn: sqlite3.Connection, query: str,
                top_k: int = 50) -> List[RetrievalResult]:
    """FTS5-powered BM25 search for broad keyword recall."""
    terms = tokenize(query)
    if not terms:
        return []

    fts_query = " OR ".join(terms)
    try:
        rows = conn.execute(
            """SELECT rowid, chunk_text, title, source_type, content_record_id
               FROM chunks_fts
               WHERE chunks_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (fts_query, top_k),
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    if not rows:
        return []

    # Batch fetch metadata
    record_ids = list(set(r[4] for r in rows if r[4]))
    meta_map = _batch_fetch_metadata(conn, record_ids)

    results = []
    for chunk_id, text, title, source, record_id in rows:
        if not text or len(text) < 20:
            continue
        meta = meta_map.get(record_id, {})
        text_lower = text.lower()

        # Score: term frequency + match ratio
        matched = [t for t in terms if t in text_lower]
        if not matched:
            continue
        match_ratio = len(matched) / max(len(terms), 1)
        tf_score = sum(min(text_lower.count(t) * 0.5, 2.0) for t in matched)
        bm25_score = match_ratio * 3.0 + tf_score

        results.append(RetrievalResult(
            chunk_id=chunk_id,
            record_id=record_id or 0,
            text=text[:500],
            title=title or meta.get("title", "Untitled"),
            source=source or meta.get("source_type", "unknown"),
            category=meta.get("category", "unknown"),
            created_at=str(meta.get("created_at", "unknown")),
            bm25_score=bm25_score,
            matched_terms=sorted(set(matched)),
        ))

    return results


# ─── Stage 2: Vector similarity ─────────────────────────────────────────

def vector_search(conn: sqlite3.Connection, query_embedding: List[float],
                  top_k: int = 50) -> List[RetrievalResult]:
    """Vector similarity search using sqlite-vec."""
    try:
        import json as _json
        rows = conn.execute(
            """SELECT ce.chunk_id, vec_distance_L2(ce.embedding, ?) as distance
               FROM chunk_embeddings ce
               ORDER BY distance
               LIMIT ?""",
            (_json.dumps(query_embedding), top_k),
        ).fetchall()
    except Exception:
        return []

    if not rows:
        return []

    chunk_ids = [r[0] for r in rows]
    distance_map = {r[0]: r[1] for r in rows}

    # Fetch chunk details
    placeholders = ",".join("?" * len(chunk_ids))
    chunks = conn.execute(
        f"""SELECT cc.id, cc.chunk_text, cc.content_record_id,
                   cr.title, cr.category, cr.created_at, cr.importance_score,
                   sf.source_type
            FROM content_chunks cc
            JOIN content_records cr ON cc.content_record_id = cr.id
            JOIN source_files sf ON cr.source_file_id = sf.id
            WHERE cc.id IN ({placeholders})""",
        chunk_ids,
    ).fetchall()

    results = []
    for c in chunks:
        chunk_id, text, record_id, title, category, created_at, importance, source = c
        distance = distance_map.get(chunk_id, 999.0)
        # Convert distance to similarity score (lower distance = higher score)
        vec_score = max(0, 10.0 - distance)

        results.append(RetrievalResult(
            chunk_id=chunk_id,
            record_id=record_id,
            text=(text or "")[:500],
            title=title or "Untitled",
            source=source or "unknown",
            category=category or "unknown",
            created_at=str(created_at) if created_at else "unknown",
            vector_score=vec_score,
        ))

    return results


# ─── Stage 3: Merge + deduplicate ───────────────────────────────────────

def merge_results(bm25_results: List[RetrievalResult],
                  vector_results: List[RetrievalResult],
                  bm25_weight: float = 0.6,
                  vector_weight: float = 0.4) -> List[RetrievalResult]:
    """Merge BM25 and vector results with weighted scoring.

    Deduplicates by chunk_id, keeping the higher-scored version.
    Combines scores from both methods when a chunk appears in both.
    """
    merged: Dict[int, RetrievalResult] = {}

    for r in bm25_results:
        r.score = r.bm25_score * bm25_weight
        merged[r.chunk_id] = r

    for r in vector_results:
        if r.chunk_id in merged:
            # Chunk found by both methods — boost it
            existing = merged[r.chunk_id]
            existing.vector_score = r.vector_score
            existing.score = (existing.bm25_score * bm25_weight +
                              r.vector_score * vector_weight)
            # Boost for appearing in both (mutual reinforcement)
            existing.score *= 1.3
        else:
            r.score = r.vector_score * vector_weight
            merged[r.chunk_id] = r

    # Sort by combined score
    results = sorted(merged.values(), key=lambda x: x.score, reverse=True)
    return results


def deduplicate_results(results: List[RetrievalResult],
                        threshold: float = 0.8) -> List[RetrievalResult]:
    """Remove near-duplicate passages based on text overlap."""
    unique = []
    seen_texts = []

    for r in results:
        text = r.text[:200].lower()
        is_dup = False
        words_a = set(text.split())
        for seen in seen_texts:
            words_b = set(seen.split())
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


# ─── Stage 4: Cross-encoder reranking ───────────────────────────────────

_reranker = None

def _load_reranker():
    global _reranker
    if _reranker is not None:
        return _reranker if _reranker != "none" else None

    try:
        import warnings
        warnings.filterwarnings('ignore')
        import os
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return _reranker
    except Exception as e:
        print(f"  Reranker unavailable: {e}")
        _reranker = "none"
        return None


def rerank(query: str, candidates: List[RetrievalResult],
           top_k: int = 10) -> List[RetrievalResult]:
    """Rerank using cross-encoder for precision boost.

    Solves the 'needle in haystack' problem: by presenting the model
    with fewer, more relevant passages, attention is focused on signal.
    """
    model = _load_reranker()
    if model is None or not candidates:
        return candidates[:top_k]

    pairs = [(query, c.text) for c in candidates]
    try:
        scores = model.predict(pairs, show_progress_bar=False)
    except Exception:
        return candidates[:top_k]

    for i, score in enumerate(scores):
        candidates[i].reranker_score = float(score)

    # Rerank by cross-encoder score, keeping original as tiebreaker
    candidates.sort(key=lambda x: (x.reranker_score, x.score), reverse=True)
    return candidates[:top_k]


# ─── Stage 5: Final scoring with recency + importance ──────────────────

def final_scoring(results: List[RetrievalResult],
                  recency_decay: int = 365) -> List[RetrievalResult]:
    """Apply recency weighting and importance boosting to final results."""
    for r in results:
        recency = recency_weight(r.created_at, recency_decay)

        # Composite final score
        if r.reranker_score > 0:
            # If reranked, trust the reranker more
            r.score = (r.reranker_score * 0.5 +
                       r.score * 0.3 +
                       recency * 1.0)
        else:
            r.score = r.score + recency * 0.8

    results.sort(key=lambda x: x.score, reverse=True)
    return results


# ─── Metadata helpers ───────────────────────────────────────────────────

def _batch_fetch_metadata(conn: sqlite3.Connection,
                          record_ids: List[int]) -> Dict[int, dict]:
    """Batch fetch metadata for record IDs."""
    if not record_ids:
        return {}

    placeholders = ",".join("?" * len(record_ids))
    rows = conn.execute(
        f"""SELECT cr.id, cr.created_at, cr.category, cr.importance_score,
                   cr.title, sf.source_type
            FROM content_records cr
            JOIN source_files sf ON cr.source_file_id = sf.id
            WHERE cr.id IN ({placeholders})""",
        record_ids,
    ).fetchall()

    return {
        r[0]: {
            "created_at": r[1],
            "category": r[2],
            "importance": r[3],
            "title": r[4],
            "source_type": r[5],
        }
        for r in rows
    }


# ─── Embedding generation ───────────────────────────────────────────────

def generate_query_embedding(query: str) -> Optional[List[float]]:
    """Generate embedding for a query using sentence-transformers or fallback."""
    try:
        import warnings
        warnings.filterwarnings('ignore')
        import os
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = model.encode(query, show_progress_bar=False)
        return embedding.tolist()
    except ImportError:
        return _hash_embedding(query)
    except Exception:
        return _hash_embedding(query)


def _hash_embedding(text: str, dimensions: int = 384) -> List[float]:
    """Simple hash-based embedding fallback."""
    import hashlib
    embedding = [0.0] * dimensions
    words = text.lower().split()[:100]
    for i, word in enumerate(words):
        h = hashlib.md5(word.encode()).hexdigest()
        for j in range(0, len(h) - 1, 2):
            pos = int(h[j:j+2], 16) % dimensions
            val = int(h[j:j+2], 16) / 255.0
            embedding[pos] += val * (1.0 - i * 0.01)
    magnitude = sum(x**2 for x in embedding) ** 0.5
    if magnitude > 0:
        embedding = [x / magnitude for x in embedding]
    return embedding


# ─── Main retrieval pipeline ────────────────────────────────────────────

def hybrid_retrieve(query: str, top_k: int = 10,
                    db_path: Path = DB_PATH,
                    use_reranker: bool = False,
                    use_vectors: bool = True,
                    strategy: str = "auto",
                    fast_mode: bool = True) -> Tuple[List[RetrievalResult], RetrievalStats]:
    """
    Full hybrid retrieval pipeline.

    Strategy options:
      - "auto": use BM25 + vectors if available, merge, optional rerank
      - "bm25_only": skip vector search (fastest)
      - "vector_only": skip BM25 (semantic-only)
      - "long_context": return more results with less filtering (for bounded sets)

    fast_mode=True (default): skips vector search unless explicitly requested.
    Vector search requires sentence-transformers model loading which is slow on CPU.
    """
    t0 = time.perf_counter()
    stats = RetrievalStats(query=query)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA cache_size=-32000")
    try:
        # Auto strategy: check what's available
        if strategy == "auto":
            has_fts5 = _check_fts5(conn)
            # In fast mode, skip vector search unless explicitly enabled
            has_vectors = (use_vectors and not fast_mode and _check_vectors(conn))
            if has_fts5 and has_vectors:
                strategy = "hybrid"
            elif has_fts5:
                strategy = "bm25_only"
            elif has_vectors:
                strategy = "vector_only"
            else:
                strategy = "bm25_only"
        elif fast_mode and strategy in ("hybrid", "long_context"):
            # fast_mode overrides: downgrade to bm25_only unless user explicitly wants vectors
            if _check_fts5(conn):
                strategy = "bm25_only"
        stats.strategy = strategy

        # Stage 1: BM25
        bm25_results = []
        if strategy in ("hybrid", "bm25_only", "long_context"):
            candidate_k = top_k * 5 if strategy == "long_context" else top_k * 3
            for variant in expand_query(query):
                bm25_results.extend(bm25_search(conn, variant, candidate_k))
            # Deduplicate BM25 results
            seen = set()
            unique_bm25 = []
            for r in bm25_results:
                if r.chunk_id not in seen:
                    seen.add(r.chunk_id)
                    unique_bm25.append(r)
            bm25_results = unique_bm25
        stats.bm25_candidates = len(bm25_results)

        # Stage 2: Vector search
        vector_results = []
        if strategy in ("hybrid", "vector_only"):
            query_embedding = generate_query_embedding(query)
            if query_embedding:
                vector_results = vector_search(conn, query_embedding, top_k * 3)
        stats.vector_candidates = len(vector_results)

        # Stage 3: Merge
        if strategy == "hybrid" and bm25_results and vector_results:
            results = merge_results(bm25_results, vector_results)
        elif strategy == "vector_only":
            results = vector_results
        else:
            results = bm25_results

        results = deduplicate_results(results)
        stats.merged_candidates = len(results)

        # Stage 4: Rerank (skip in fast mode / bm25_only)
        if use_reranker and strategy not in ("bm25_only",) and len(results) > top_k:
            results = rerank(query, results, top_k * 2)
            stats.reranked = True

        # Stage 5: Final scoring
        results = final_scoring(results)
        results = results[:top_k]
        stats.final_count = len(results)

    finally:
        conn.close()

    stats.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return results, stats


def _check_fts5(conn: sqlite3.Connection) -> bool:
    """Check if FTS5 index is available and populated."""
    try:
        cnt = conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
        return cnt > 0
    except Exception:
        return False


def _check_vectors(conn: sqlite3.Connection) -> bool:
    """Check if vector embeddings are available."""
    try:
        cnt = conn.execute("SELECT count(*) FROM chunk_embeddings").fetchone()[0]
        return cnt > 0
    except Exception:
        return False


# ─── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hybrid retrieval pipeline")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--no-vectors", action="store_true")
    parser.add_argument("--strategy", default="auto",
                        choices=["auto", "bm25_only", "vector_only", "long_context"])
    args = parser.parse_args()

    if args.query:
        results, stats = hybrid_retrieve(
            args.query, args.top_k, args.db,
            use_reranker=args.rerank,
            use_vectors=not args.no_vectors,
            strategy=args.strategy,
        )
        print(f"\n{'='*60}")
        print(f"Query: {args.query}")
        print(f"Strategy: {stats.strategy}")
        print(f"Candidates: BM25={stats.bm25_candidates} Vec={stats.vector_candidates} "
              f"Merged={stats.merged_candidates} Final={stats.final_count}")
        print(f"Reranked: {stats.reranked}")
        print(f"Time: {stats.elapsed_ms}ms")
        print(f"{'='*60}\n")

        for i, r in enumerate(results, 1):
            print(f"{i}. [{r.source}] score={r.score:.2f} "
                  f"(bm25={r.bm25_score:.2f} vec={r.vector_score:.2f} "
                  f"rerank={r.reranker_score:.2f})")
            print(f"   {r.title[:60]}")
            print(f"   {r.text[:150]}...")
            if r.matched_terms:
                print(f"   Terms: {', '.join(r.matched_terms)}")
            print()
    else:
        print("Usage: python retriever.py 'your query' [--rerank] [--strategy auto|bm25_only|vector_only|long_context]")
