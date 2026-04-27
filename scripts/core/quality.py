#!/usr/bin/env python3
"""
Knowledge quality metrics for LocalBrain.

Tracks retrieval quality and detects "silent failures" — the key risk
identified in the video: "The answer existed in the data, but the LLM
never saw it because the retrieval step didn't return the right results."

Metrics:
  - Retrieval coverage: what % of relevant docs are found
  - Query success rate: how often users get satisfactory answers
  - Embedding quality: how well vectors capture semantic meaning
  - Chunk overlap: how much information is lost at boundaries
  - Retrieval latency: performance tracking
  - Silent failure detection: identify when retrieval misses good data
"""

import json
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "state.db"


@dataclass
class RetrievalMetric:
    """Metrics from a single retrieval operation."""
    query: str
    query_type: str = "unknown"
    strategy: str = "unknown"
    bm25_candidates: int = 0
    vector_candidates: int = 0
    final_count: int = 0
    elapsed_ms: float = 0.0
    top_score: float = 0.0
    avg_score: float = 0.0
    score_spread: float = 0.0  # gap between best and worst result
    sources_count: int = 0     # how many distinct sources in results
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class QualityReport:
    """Summary quality report."""
    total_queries: int = 0
    avg_latency_ms: float = 0.0
    avg_results_per_query: float = 0.0
    zero_result_rate: float = 0.0      # % of queries with no results
    low_confidence_rate: float = 0.0   # % of queries with top_score < threshold
    source_diversity: float = 0.0      # avg distinct sources per query
    strategy_breakdown: Dict[str, int] = field(default_factory=dict)
    silent_failure_candidates: int = 0  # queries where retrieval may have missed

    def to_dict(self) -> dict:
        return asdict(self)


class QualityTracker:
    """Track and analyze retrieval quality over time."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._metrics: List[RetrievalMetric] = []
        self._ensure_table()

    def _ensure_table(self):
        """Create metrics table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS retrieval_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    query_type TEXT,
                    strategy TEXT,
                    bm25_candidates INTEGER,
                    vector_candidates INTEGER,
                    final_count INTEGER,
                    elapsed_ms REAL,
                    top_score REAL,
                    avg_score REAL,
                    score_spread REAL,
                    sources_count INTEGER,
                    timestamp TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp
                ON retrieval_metrics(timestamp)
            """)
            conn.commit()
        finally:
            conn.close()

    def record(self, metric: RetrievalMetric):
        """Record a retrieval metric."""
        self._metrics.append(metric)

        # Persist to DB
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO retrieval_metrics
                (query, query_type, strategy, bm25_candidates, vector_candidates,
                 final_count, elapsed_ms, top_score, avg_score, score_spread,
                 sources_count, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.query, metric.query_type, metric.strategy,
                metric.bm25_candidates, metric.vector_candidates,
                metric.final_count, metric.elapsed_ms, metric.top_score,
                metric.avg_score, metric.score_spread, metric.sources_count,
                metric.timestamp,
            ))
            conn.commit()
        finally:
            conn.close()

    def report(self, days: int = 7) -> QualityReport:
        """Generate quality report for the last N days."""
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute("""
                SELECT query, query_type, strategy, bm25_candidates,
                       vector_candidates, final_count, elapsed_ms,
                       top_score, avg_score, score_spread, sources_count
                FROM retrieval_metrics
                WHERE timestamp >= datetime('now', ?)
                ORDER BY timestamp DESC
            """, (f"-{days} days",)).fetchall()
        finally:
            conn.close()

        if not rows:
            return QualityReport()

        total = len(rows)
        zero_results = sum(1 for r in rows if r[5] == 0)
        low_conf = sum(1 for r in rows if r[7] < 2.0)  # top_score < 2.0
        latencies = [r[6] for r in rows]
        result_counts = [r[5] for r in rows]
        source_counts = [r[10] for r in rows]
        strategy_counts = defaultdict(int)
        for r in rows:
            strategy_counts[r[2] or "unknown"] += 1

        # Silent failure heuristic: queries where:
        #   - BM25 found candidates but final count is low (dedup killed results)
        #   - OR top_score is very low (weak matches)
        silent_failures = sum(
            1 for r in rows
            if (r[3] > 0 and r[5] < 2) or  # BM25 had candidates but barely any results
            (r[7] > 0 and r[7] < 1.0)       # Very low confidence scores
        )

        return QualityReport(
            total_queries=total,
            avg_latency_ms=sum(latencies) / max(total, 1),
            avg_results_per_query=sum(result_counts) / max(total, 1),
            zero_result_rate=zero_results / max(total, 1),
            low_confidence_rate=low_conf / max(total, 1),
            source_diversity=sum(source_counts) / max(total, 1),
            strategy_breakdown=dict(strategy_counts),
            silent_failure_candidates=silent_failures,
        )

    def get_recent_slow_queries(self, threshold_ms: float = 2000,
                                 limit: int = 10) -> List[dict]:
        """Find queries that took too long."""
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute("""
                SELECT query, strategy, elapsed_ms, final_count, timestamp
                FROM retrieval_metrics
                WHERE elapsed_ms > ?
                ORDER BY elapsed_ms DESC
                LIMIT ?
            """, (threshold_ms, limit)).fetchall()

            return [
                {
                    "query": r[0],
                    "strategy": r[1],
                    "elapsed_ms": r[2],
                    "result_count": r[3],
                    "timestamp": r[4],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def get_zero_result_queries(self, limit: int = 20) -> List[dict]:
        """Find queries that returned no results (retrieval failures)."""
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute("""
                SELECT query, query_type, strategy, bm25_candidates,
                       vector_candidates, timestamp
                FROM retrieval_metrics
                WHERE final_count = 0
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,)).fetchall()

            return [
                {
                    "query": r[0],
                    "query_type": r[1],
                    "strategy": r[2],
                    "bm25_had_candidates": r[3] > 0,
                    "vector_had_candidates": r[4] > 0,
                    "timestamp": r[5],
                }
                for r in rows
            ]
        finally:
            conn.close()


def compute_index_stats(db_path: Path = DB_PATH) -> dict:
    """Compute statistics about the knowledge index quality."""
    conn = sqlite3.connect(db_path)
    try:
        stats = {}

        # Total records and chunks
        stats["total_records"] = conn.execute(
            "SELECT COUNT(*) FROM content_records"
        ).fetchone()[0]
        stats["total_chunks"] = conn.execute(
            "SELECT COUNT(*) FROM content_chunks"
        ).fetchone()[0]

        # Average chunk size
        row = conn.execute(
            "SELECT AVG(LENGTH(chunk_text)), MIN(LENGTH(chunk_text)), MAX(LENGTH(chunk_text))"
            " FROM content_chunks"
        ).fetchone()
        stats["avg_chunk_chars"] = round(row[0] or 0)
        stats["min_chunk_chars"] = row[1] or 0
        stats["max_chunk_chars"] = row[2] or 0

        # Source distribution
        rows = conn.execute("""
            SELECT sf.source_type, COUNT(DISTINCT cr.id)
            FROM content_records cr
            JOIN source_files sf ON cr.source_file_id = sf.id
            GROUP BY sf.source_type
            ORDER BY COUNT(*) DESC
        """).fetchall()
        stats["source_distribution"] = {r[0]: r[1] for r in rows}

        # Category distribution
        rows = conn.execute("""
            SELECT category, COUNT(*) FROM content_records
            GROUP BY category ORDER BY COUNT(*) DESC
        """).fetchall()
        stats["category_distribution"] = {r[0]: r[1] for r in rows}

        # Embedding coverage
        embedded = conn.execute(
            "SELECT COUNT(*) FROM content_chunks WHERE embedding_status = 'embedded'"
        ).fetchone()[0]
        stats["embedding_coverage"] = embedded / max(stats["total_chunks"], 1)
        stats["embedded_chunks"] = embedded

        # FTS5 coverage
        try:
            fts5_count = conn.execute(
                "SELECT COUNT(*) FROM chunks_fts"
            ).fetchone()[0]
            stats["fts5_coverage"] = fts5_count / max(stats["total_chunks"], 1)
            stats["fts5_entries"] = fts5_count
        except Exception:
            stats["fts5_coverage"] = 0.0
            stats["fts5_entries"] = 0

        # Records per day (recent activity)
        rows = conn.execute("""
            SELECT DATE(created_at), COUNT(*)
            FROM content_records
            WHERE created_at >= date('now', '-30 days')
            GROUP BY DATE(created_at)
            ORDER BY DATE(created_at) DESC
            LIMIT 30
        """).fetchall()
        stats["daily_activity"] = {r[0]: r[1] for r in rows}

        return stats

    finally:
        conn.close()


# ─── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Knowledge quality metrics")
    parser.add_argument("--report", action="store_true", help="Show quality report")
    parser.add_argument("--index-stats", action="store_true", help="Show index statistics")
    parser.add_argument("--failures", action="store_true", help="Show zero-result queries")
    parser.add_argument("--days", type=int, default=7, help="Report period in days")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()

    if args.index_stats or not any([args.report, args.failures]):
        stats = compute_index_stats(args.db)
        print("\n=== Knowledge Index Statistics ===")
        print(f"Records: {stats.get('total_records', 0)}")
        print(f"Chunks: {stats.get('total_chunks', 0)}")
        print(f"Chunk size: avg={stats.get('avg_chunk_chars', 0)} "
              f"min={stats.get('min_chunk_chars', 0)} max={stats.get('max_chunk_chars', 0)}")
        print(f"FTS5: {stats.get('fts5_entries', 0)} entries "
              f"({stats.get('fts5_coverage', 0):.1%} coverage)")
        print(f"Embeddings: {stats.get('embedded_chunks', 0)} "
              f"({stats.get('embedding_coverage', 0):.1%} coverage)")
        print(f"\nSources:")
        for src, cnt in stats.get("source_distribution", {}).items():
            print(f"  {src}: {cnt}")
        print(f"\nCategories:")
        for cat, cnt in stats.get("category_distribution", {}).items():
            print(f"  {cat}: {cnt}")

    if args.report:
        tracker = QualityTracker(args.db)
        report = tracker.report(args.days)
        print(f"\n=== Quality Report (last {args.days} days) ===")
        print(f"Total queries: {report.total_queries}")
        print(f"Avg latency: {report.avg_latency_ms:.0f}ms")
        print(f"Avg results/query: {report.avg_results_per_query:.1f}")
        print(f"Zero-result rate: {report.zero_result_rate:.1%}")
        print(f"Low-confidence rate: {report.low_confidence_rate:.1%}")
        print(f"Source diversity: {report.source_diversity:.1f} sources/query")
        print(f"Silent failure candidates: {report.silent_failure_candidates}")
        print(f"Strategies used: {report.strategy_breakdown}")

    if args.failures:
        tracker = QualityTracker(args.db)
        failures = tracker.get_zero_result_queries()
        if failures:
            print(f"\n=== Zero-Result Queries ===")
            for f in failures:
                print(f"  [{f['query_type']}] {f['query'][:60]}...")
                print(f"    strategy={f['strategy']} bm25={f['bm25_had_candidates']} "
                      f"vec={f['vector_had_candidates']} at {f['timestamp']}")
        else:
            print("\nNo zero-result queries recorded.")
