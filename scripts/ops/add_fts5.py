#!/usr/bin/env python3
"""
Add FTS5 full-text search index on content_chunks.chunk_text.
This enables fast keyword search without loading all chunks into memory.
Run once to create and populate the FTS5 table.
"""
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "state.db"


def add_fts5(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    # Check if FTS5 table already exists
    try:
        conn.execute("SELECT count(*) FROM chunks_fts")
        print("FTS5 table already exists. Checking if populated...")
        cnt = conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
        if cnt > 0:
            print(f"FTS5 index already has {cnt} entries. Nothing to do.")
            conn.close()
            return
    except sqlite3.OperationalError:
        pass

    # Create FTS5 virtual table
    print("Creating FTS5 virtual table...")
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_text,
            title,
            source_type,
            content_record_id,
            content='content_chunks',
            tokenize='unicode61'
        )
    """)
    conn.commit()

    # Populate FTS5 from existing chunks
    print("Populating FTS5 index (this may take a minute for large databases)...")
    t0 = time.time()

    # Use batch inserts for efficiency
    batch_size = 5000
    cursor = conn.execute("""
        SELECT cc.id, cc.chunk_text, cr.title, sf.source_type, cc.content_record_id
        FROM content_chunks cc
        JOIN content_records cr ON cc.content_record_id = cr.id
        JOIN source_files sf ON cr.source_file_id = sf.id
    """)

    total = 0
    batch = []
    for row in cursor:
        chunk_id, chunk_text, title, source_type, record_id = row
        # Clean text for FTS5
        clean_text = (chunk_text or "").replace("\n", " ").replace("\r", " ")[:2000]
        batch.append((chunk_id, clean_text, title or "", source_type or "", record_id))
        total += 1

        if len(batch) >= batch_size:
            conn.executemany(
                "INSERT OR IGNORE INTO chunks_fts (rowid, chunk_text, title, source_type, content_record_id) VALUES (?, ?, ?, ?, ?)",
                batch,
            )
            conn.commit()
            print(f"  Inserted {total} rows...")
            batch = []

    if batch:
        conn.executemany(
            "INSERT OR IGNORE INTO chunks_fts (rowid, chunk_text, title, source_type, content_record_id) VALUES (?, ?, ?, ?, ?)",
            batch,
        )
        conn.commit()

    elapsed = time.time() - t0
    print(f"FTS5 index populated: {total} entries in {elapsed:.1f}s")

    conn.close()


if __name__ == "__main__":
    add_fts5()
