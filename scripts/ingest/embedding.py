#!/usr/bin/env python3
"""
Real embedding pipeline for LocalBrain.

Upgrades from hash-based fake embeddings to real sentence-transformer embeddings.
Uses sentence-transformers (all-MiniLM-L6-v2) for GPU-accelerated embedding generation.
Falls back to hash-based embeddings if sentence-transformers is unavailable.

The video emphasizes that embeddings are the foundation of semantic search quality.
Good embeddings → good vector retrieval → fewer silent failures.
"""

import json
import sqlite3
import sys
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "state.db"

# Suppress noisy logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

EMBEDDING_DIM = 384
DEFAULT_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64

# Lazy-loaded model
_embedding_model = None


def get_embedding_model():
    """Load sentence-transformers model (lazy, once)."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    try:
        import warnings
        warnings.filterwarnings('ignore')
        import os
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {DEFAULT_MODEL}...")
        _embedding_model = SentenceTransformer(DEFAULT_MODEL)
        print(f"  Model loaded: {DEFAULT_MODEL} (dim={EMBEDDING_DIM})")
        # Check for GPU
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                print(f"  GPU: {device_name}")
            else:
                print("  Running on CPU (install torch+cuda for GPU acceleration)")
        except ImportError:
            pass
        return _embedding_model
    except ImportError:
        print("WARNING: sentence-transformers not available, using hash-based fallback")
        _embedding_model = "fallback"
        return "fallback"
    except Exception as e:
        print(f"WARNING: Could not load embedding model: {e}")
        _embedding_model = "fallback"
        return "fallback"


def generate_embedding(text: str) -> List[float]:
    """Generate embedding for a single text."""
    model = get_embedding_model()

    if model == "fallback":
        return _hash_embedding(text)

    try:
        embedding = model.encode(text[:1000], show_progress_bar=False)
        return embedding.tolist()
    except Exception as e:
        print(f"  Embedding error: {e}")
        return _hash_embedding(text)


def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a batch of texts (much faster than one-by-one)."""
    model = get_embedding_model()

    if model == "fallback":
        return [_hash_embedding(t) for t in texts]

    try:
        # Truncate long texts
        truncated = [t[:1000] for t in texts]
        embeddings = model.encode(truncated, batch_size=BATCH_SIZE,
                                   show_progress_bar=False)
        return embeddings.tolist()
    except Exception as e:
        print(f"  Batch embedding error: {e}")
        return [_hash_embedding(t) for t in texts]


def _hash_embedding(text: str, dimensions: int = EMBEDDING_DIM) -> List[float]:
    """Hash-based embedding fallback (not as good as neural, but functional)."""
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


def init_vector_tables(conn: sqlite3.Connection) -> bool:
    """Initialize vector storage tables."""
    # Try sqlite-vec first
    try:
        import sqlite_vec
        sqlite_vec.load(conn)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunk_embeddings'"
        )
        if not cursor.fetchone():
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings USING vec0(
                    chunk_id INTEGER PRIMARY KEY,
                    embedding FLOAT[{EMBEDDING_DIM}]
                )
            """)
            print("  Created chunk_embeddings (sqlite-vec)")
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"  sqlite-vec unavailable: {e}")

    # Fallback table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunk_embeddings_fallback (
            chunk_id INTEGER PRIMARY KEY,
            embedding_model TEXT,
            embedding_data BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chunk_id) REFERENCES content_chunks(id)
        )
    """)
    print("  Using fallback embedding table")
    return False


def run_embedding(db_path: Path = DB_PATH, batch_size: int = BATCH_SIZE,
                  limit: int = 0) -> dict:
    """Generate embeddings for all pending chunks.

    Args:
        db_path: Path to database
        batch_size: Number of chunks to embed per batch
        limit: Max chunks to process (0 = all)
    """
    conn = sqlite3.connect(db_path)

    try:
        stats = {
            'started_at': datetime.now().isoformat(),
            'chunks_embedded': 0,
            'batches_processed': 0,
            'errors': [],
        }

        vec_ready = init_vector_tables(conn)

        # Get pending chunks
        sql = """
            SELECT cc.id, cc.chunk_text
            FROM content_chunks cc
            WHERE cc.embedding_status = 'pending'
            ORDER BY cc.id
        """
        if limit > 0:
            sql += f" LIMIT {limit}"

        pending = conn.execute(sql).fetchall()

        if not pending:
            print("No pending chunks to embed")
            return stats

        print(f"Found {len(pending)} pending chunks to embed")

        # Process in batches
        total = len(pending)
        for i in range(0, total, batch_size):
            batch = pending[i:i+batch_size]
            batch_texts = [row[1] for row in batch]
            batch_ids = [row[0] for row in batch]

            t0 = time.perf_counter()

            try:
                embeddings = generate_embeddings_batch(batch_texts)
            except Exception as e:
                print(f"  Batch {stats['batches_processed']+1} failed: {e}")
                # Mark as failed
                for chunk_id in batch_ids:
                    conn.execute(
                        "UPDATE content_chunks SET embedding_status = 'failed' WHERE id = ?",
                        (chunk_id,)
                    )
                stats['errors'].append(str(e))
                stats['batches_processed'] += 1
                continue

            # Store embeddings
            model_name = DEFAULT_MODEL if get_embedding_model() != "fallback" else "hash_fallback"

            for chunk_id, embedding in zip(batch_ids, embeddings):
                try:
                    if vec_ready:
                        conn.execute(
                            "INSERT OR REPLACE INTO chunk_embeddings (chunk_id, embedding) VALUES (?, ?)",
                            (chunk_id, json.dumps(embedding))
                        )
                    else:
                        import struct
                        emb_blob = struct.pack(f'{len(embedding)}f', *embedding)
                        conn.execute(
                            "INSERT OR REPLACE INTO chunk_embeddings_fallback (chunk_id, embedding_model, embedding_data) VALUES (?, ?, ?)",
                            (chunk_id, model_name, emb_blob)
                        )

                    conn.execute(
                        "UPDATE content_chunks SET embedding_status = 'embedded', embedding_model = ? WHERE id = ?",
                        (model_name, chunk_id)
                    )
                    stats['chunks_embedded'] += 1
                except Exception as e:
                    stats['errors'].append(f"chunk {chunk_id}: {e}")
                    conn.execute(
                        "UPDATE content_chunks SET embedding_status = 'failed' WHERE id = ?",
                        (chunk_id,)
                    )

            stats['batches_processed'] += 1
            elapsed = time.perf_counter() - t0

            conn.commit()
            done = min(i + batch_size, total)
            rate = len(batch) / max(elapsed, 0.001)
            print(f"  Batch {stats['batches_processed']}: {done}/{total} chunks "
                  f"({rate:.0f} chunks/sec, {elapsed:.1f}s)")

        stats['completed_at'] = datetime.now().isoformat()

        print(f"\nEmbedding complete:")
        print(f"  Chunks embedded: {stats['chunks_embedded']}")
        print(f"  Batches: {stats['batches_processed']}")
        print(f"  Errors: {len(stats['errors'])}")
        print(f"  sqlite-vec: {vec_ready}")

        return stats

    finally:
        conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate embeddings for chunks")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=0, help="Max chunks (0=all)")
    parser.add_argument("--test", type=str, help="Test embedding for text")

    args = parser.parse_args()

    if args.test:
        emb = generate_embedding(args.test)
        print(f"Embedding dim: {len(emb)}")
        print(f"First 5 values: {emb[:5]}")
        print(f"Magnitude: {sum(x**2 for x in emb)**0.5:.4f}")
    else:
        result = run_embedding(args.db, args.batch_size, args.limit)
