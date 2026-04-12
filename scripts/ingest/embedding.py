#!/usr/bin/env python3
"""
Generate embeddings for content chunks using sqlite-vec.
Falls back to simple keyword-based search if sqlite-vec is not available.
Optimized for batch processing with checkpointing.
"""

import json
import sqlite3
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "state.db"

# Try to import sqlite-vec
try:
    import sqlite_vec
    SQLITE_VEC_AVAILABLE = True
    print("sqlite-vec loaded successfully")
except ImportError:
    SQLITE_VEC_AVAILABLE = False
    print("WARNING: sqlite-vec not available, using fallback mode")

# Embedding dimensions (using all-MiniLM-L6-v2: 384 dimensions)
EMBEDDING_DIM = 384


def init_vector_tables(conn: sqlite3.Connection):
    """Initialize sqlite-vec virtual tables if available."""
    if not SQLITE_VEC_AVAILABLE:
        return False
    
    try:
        # Load sqlite-vec extension
        sqlite_vec.load(conn)
        
        # Check if table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunk_embeddings'"
        )
        if cursor.fetchone():
            print("  chunk_embeddings table already exists")
            return True
        
        # Create virtual table for embeddings
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding FLOAT[{EMBEDDING_DIM}]
            )
        """)
        
        print("  Created chunk_embeddings virtual table")
        return True
        
    except Exception as e:
        print(f"  WARNING: Could not initialize sqlite-vec: {e}")
        return False


def generate_simple_embedding(text: str) -> list[float]:
    """Generate a simple bag-of-words embedding as fallback.
    Not as good as neural embeddings, but works without heavy dependencies.
    """
    # Simple character-level embedding
    embedding = [0.0] * EMBEDDING_DIM
    
    # Normalize text
    text = text.lower()[:1000]  # Limit length
    
    # Create a simple hash-based embedding
    words = text.split()
    for i, word in enumerate(words[:100]):  # Limit words
        for j, char in enumerate(word[:20]):
            idx = (i * 3 + j * 7 + ord(char)) % EMBEDDING_DIM
            embedding[idx] += 1.0
    
    # Normalize
    magnitude = sum(x**2 for x in embedding) ** 0.5
    if magnitude > 0:
        embedding = [x / magnitude for x in embedding]
    
    return embedding


def generate_embedding(text: str) -> list[float]:
    """Generate embedding for text.
    Currently uses simple fallback. Can be upgraded to use sentence-transformers or local model.
    """
    # For now, use simple embedding
    # TODO: Integrate with local llama.cpp for embedding generation
    return generate_simple_embedding(text)


def run_embedding(db_path: Path = DB_PATH, batch_size: int = 100) -> dict:
    """Generate embeddings for all pending chunks."""
    
    conn = sqlite3.connect(db_path)
    
    try:
        stats = {
            'started_at': datetime.now().isoformat(),
            'chunks_embedded': 0,
            'batches_processed': 0,
            'errors': [],
            'sqlite_vec_available': SQLITE_VEC_AVAILABLE
        }
        
        # Initialize vector tables
        vec_ready = init_vector_tables(conn)
        stats['vector_table_ready'] = vec_ready
        
        # Get pending chunks
        pending = conn.execute(
            """SELECT cc.id, cc.chunk_text, cr.category, cr.importance_score
               FROM content_chunks cc
               JOIN content_records cr ON cc.content_record_id = cr.id
               WHERE cc.embedding_status = 'pending'
               LIMIT ?""",
            (batch_size * 10,)  # Get more to process in batches
        ).fetchall()
        
        if not pending:
            print("No pending chunks to embed")
            return stats
        
        print(f"Found {len(pending)} pending chunks to embed")
        
        # Process in batches
        for i in range(0, len(pending), batch_size):
            batch = pending[i:i+batch_size]
            
            for chunk_id, chunk_text, category, importance in batch:
                try:
                    # Generate embedding
                    embedding = generate_embedding(chunk_text)
                    
                    if SQLITE_VEC_AVAILABLE and vec_ready:
                        # Store in sqlite-vec virtual table
                        conn.execute(
                            """INSERT OR REPLACE INTO chunk_embeddings (chunk_id, embedding)
                               VALUES (?, ?)""",
                            (chunk_id, json.dumps(embedding))
                        )
                    
                    # Update chunk status
                    conn.execute(
                        """UPDATE content_chunks 
                           SET embedding_status = 'embedded',
                               embedding_model = 'simple_fallback_v1'
                           WHERE id = ?""",
                        (chunk_id,)
                    )
                    
                    stats['chunks_embedded'] += 1
                    
                except Exception as e:
                    stats['errors'].append({'chunk_id': chunk_id, 'error': str(e)})
                    
                    # Mark as failed
                    conn.execute(
                        """UPDATE content_chunks 
                           SET embedding_status = 'failed'
                           WHERE id = ?""",
                        (chunk_id,)
                    )
            
            stats['batches_processed'] += 1
            conn.commit()
            print(f"  Embedded batch {stats['batches_processed']}: {len(batch)} chunks")
        
        # Update embedding counts in batch tracking
        conn.execute(
            """UPDATE ingestion_batches 
               SET embeddings_created = (
                   SELECT COUNT(*) FROM content_chunks WHERE embedding_status = 'embedded'
               )
               WHERE id = (SELECT id FROM ingestion_batches WHERE status = 'completed' ORDER BY id DESC LIMIT 1)"""
        )
        
        stats['completed_at'] = datetime.now().isoformat()
        conn.commit()
        
        print(f"\nEmbedding complete:")
        print(f"  Chunks embedded: {stats['chunks_embedded']}")
        print(f"  Batches: {stats['batches_processed']}")
        print(f"  Errors: {len(stats['errors'])}")
        print(f"  sqlite-vec active: {vec_ready}")
        
        return stats
        
    finally:
        conn.close()


def search_similar(db_path: Path, query: str, top_k: int = 10) -> list[dict]:
    """Search for similar chunks using vector similarity."""
    
    conn = sqlite3.connect(db_path)
    
    try:
        # Generate query embedding
        query_embedding = generate_embedding(query)
        
        if SQLITE_VEC_AVAILABLE:
            try:
                sqlite_vec.load(conn)
                
                # Vector similarity search
                results = conn.execute(
                    """SELECT ce.chunk_id, vec_distance_L2(ce.embedding, ?) as distance
                       FROM chunk_embeddings ce
                       ORDER BY distance
                       LIMIT ?""",
                    (json.dumps(query_embedding), top_k)
                ).fetchall()
                
                # Get full chunk details
                chunk_ids = [r[0] for r in results]
                if chunk_ids:
                    placeholders = ','.join('?' * len(chunk_ids))
                    chunks = conn.execute(
                        f"""SELECT cc.id, cc.chunk_text, cc.chunk_index, 
                                   cr.title, cr.category, cr.importance_score, cr.id as record_id
                           FROM content_chunks cc
                           JOIN content_records cr ON cc.content_record_id = cr.id
                           WHERE cc.id IN ({placeholders})""",
                        chunk_ids
                    ).fetchall()
                    
                    return [
                        {
                            'chunk_id': c[0],
                            'text': c[1][:500],
                            'index': c[2],
                            'title': c[3],
                            'category': c[4],
                            'importance': c[5],
                            'record_id': c[6]
                        }
                        for c in chunks
                    ]
            except Exception as e:
                print(f"Vector search failed: {e}, falling back to keyword")
        
        # Fallback: keyword search
        keywords = query.lower().split()
        sql = """
            SELECT cc.id, cc.chunk_text, cc.chunk_index,
                   cr.title, cr.category, cr.importance_score, cr.id as record_id
            FROM content_chunks cc
            JOIN content_records cr ON cc.content_record_id = cr.id
            WHERE 1=1
        """
        params = []
        
        for kw in keywords[:5]:
            sql += " AND LOWER(cc.chunk_text) LIKE ?"
            params.append(f"%{kw}%")
        
        sql += " ORDER BY cr.importance_score DESC, cc.id DESC LIMIT ?"
        params.append(top_k)
        
        results = conn.execute(sql, params).fetchall()
        
        return [
            {
                'chunk_id': r[0],
                'text': r[1][:500],
                'index': r[2],
                'title': r[3],
                'category': r[4],
                'importance': r[5],
                'record_id': r[6]
            }
            for r in results
        ]
        
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings for chunks")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Database path")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--search", type=str, help="Search query (for testing)")
    parser.add_argument("--top-k", type=int, default=10)
    
    args = parser.parse_args()
    
    if args.search:
        results = search_similar(args.db, args.search, args.top_k)
        print(f"\nSearch results for: {args.search}")
        for i, r in enumerate(results, 1):
            title = r['title'] or 'Untitled'
            print(f"\n{i}. [{r['category']}] {title[:50]}...")
            print(f"   {r['text'][:200]}...")
    else:
        result = run_embedding(args.db, args.batch_size)
        sys.exit(0)
