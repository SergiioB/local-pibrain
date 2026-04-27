#!/usr/bin/env python3
"""
Chunk content records using smart strategies.

Upgrades from simple fixed-size chunking to strategy-aware chunking that
selects the best approach based on content type:
  - Conversations → turn-aware chunking
  - Structured docs → semantic grouping
  - Code → fixed-size
  - General text → recursive structure-aware

The video highlights chunking as critical RAG infrastructure.
Good chunking → better retrieval → fewer silent failures.
"""

import hashlib
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "state.db"

# Import smart chunking
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.smart_chunk import chunk_record, ChunkStrategy


def get_record_content(conn: sqlite3.Connection, record_id: int) -> tuple:
    """Extract full content from a record including source type hint.

    Returns (content_text, source_type) tuple.
    """
    record = conn.execute(
        """SELECT cr.title, cr.category, cr.metadata, cr.record_type, cr.external_id,
                  sf.path, sf.source_type
           FROM content_records cr
           JOIN source_files sf ON cr.source_file_id = sf.id
           WHERE cr.id = ?""",
        (record_id,)
    ).fetchone()

    if not record:
        return "", "unknown"

    title, category, metadata_json, record_type, external_id, source_path, source_type = record
    metadata = json.loads(metadata_json) if metadata_json else {}

    content_parts = []

    if title:
        content_parts.append(f"Title: {title}")
    if category:
        content_parts.append(f"Category: {category}")

    if metadata:
        if isinstance(metadata.get('content_text'), str) and len(metadata['content_text']) > 20:
            return metadata['content_text'], source_type

        for key in ['content', 'text', 'description', 'summary', 'body']:
            if key in metadata:
                val = metadata[key]
                if isinstance(val, str) and len(val) > 50:
                    content_parts.append(val)
                elif isinstance(val, list) and len(val) > 0:
                    content_parts.append('\n'.join(str(v) for v in val[:10]))

    # Try to read from source file if JSONL
    if source_path and (source_path.endswith('.jsonl') or 'jsonl' in str(source_path)):
        try:
            source_file = Path(source_path)
            if source_file.exists():
                with open(source_file, 'r', encoding='utf-8', errors='replace') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            line_data = json.loads(line.strip())
                            line_id = line_data.get('id') or line_data.get('conversation_id') or f"line_{line_num}"
                            if str(line_id) == str(external_id):
                                messages = line_data.get('messages') or line_data.get('turns') or []
                                msg_texts = []
                                for msg in messages:
                                    role = msg.get('role', 'unknown')
                                    content = msg.get('content') or msg.get('text', '')
                                    if isinstance(content, list):
                                        content = ' '.join(str(c) for c in content)
                                    if content:
                                        msg_texts.append(f"{role}: {content}")
                                if msg_texts:
                                    content_parts.append('\n'.join(msg_texts))
                                break
                        except:
                            continue
        except Exception as e:
            print(f"    Warning: Could not read source file: {e}")

    if metadata and 'content_preview' in metadata:
        content_parts.append(metadata['content_preview'])

    return '\n\n'.join(content_parts), source_type


def run_chunking(db_path: Path = DB_PATH, chunk_size: int = 512,
                 overlap: int = 64, strategy: str = "auto") -> dict:
    """Chunk all content records using smart strategies."""
    conn = sqlite3.connect(db_path)

    try:
        stats = {
            'started_at': datetime.now().isoformat(),
            'records_chunked': 0,
            'chunks_created': 0,
            'strategy_counts': {},
            'errors': []
        }

        # Get content records without chunks
        records = conn.execute(
            """SELECT cr.id, cr.source_file_id
               FROM content_records cr
               WHERE NOT EXISTS (
                   SELECT 1 FROM content_chunks cc WHERE cc.content_record_id = cr.id
               )"""
        ).fetchall()

        print(f"Found {len(records)} records to chunk (strategy: {strategy})")

        for i, (record_id, source_file_id) in enumerate(records):
            try:
                content_text, source_type = get_record_content(conn, record_id)

                if not content_text or len(content_text.strip()) < 50:
                    continue

                # Use smart chunking
                chunks = chunk_record(
                    content_text,
                    source_type=source_type,
                    strategy=strategy,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )

                if not chunks:
                    continue

                for chunk_data in chunks:
                    if chunk_data['text'] and len(chunk_data['text']) > 20:
                        # Track strategy usage
                        strat = chunk_data['strategy']
                        stats['strategy_counts'][strat] = stats['strategy_counts'].get(strat, 0) + 1

                        conn.execute(
                            """INSERT INTO content_chunks
                               (content_record_id, chunk_index, chunk_text, chunk_hash,
                                start_char, end_char, embedding_status, metadata)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                            (record_id, chunk_data['index'], chunk_data['text'],
                             chunk_data['hash'], chunk_data['start'], chunk_data['end'],
                             'pending', chunk_data.get('metadata', '{}'))
                        )
                        stats['chunks_created'] += 1

                stats['records_chunked'] += 1

                if stats['records_chunked'] % 50 == 0:
                    conn.commit()
                    print(f"  Chunked {stats['records_chunked']}/{len(records)} records... "
                          f"({stats['chunks_created']} chunks)")

            except Exception as e:
                stats['errors'].append({'record_id': record_id, 'error': str(e)})

        # Update source file chunk counts
        conn.execute(
            """UPDATE source_files
               SET chunk_count = (
                   SELECT COUNT(*) FROM content_chunks cc
                   JOIN content_records cr ON cc.content_record_id = cr.id
                   WHERE cr.source_file_id = source_files.id
               )"""
        )

        stats['completed_at'] = datetime.now().isoformat()
        conn.commit()

        print(f"\nChunking complete:")
        print(f"  Records chunked: {stats['records_chunked']}")
        print(f"  Chunks created: {stats['chunks_created']}")
        print(f"  Strategies used: {stats['strategy_counts']}")
        print(f"  Errors: {len(stats['errors'])}")

        return stats

    finally:
        conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smart chunking pipeline")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--strategy", default="auto",
                        choices=["auto", "fixed", "sliding", "recursive", "semantic"])

    args = parser.parse_args()
    result = run_chunking(args.db, args.chunk_size, args.overlap, args.strategy)
