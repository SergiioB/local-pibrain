#!/usr/bin/env python3
"""
Initialize the SQLite database for personal AI node.
Loads schema from schemas/metadata.sql and creates state.db.
"""

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "metadata.sql"
DB_PATH = PROJECT_ROOT / "data" / "state.db"


def init_database(db_path: Path = DB_PATH, schema_path: Path = SCHEMA_PATH) -> bool:
    """Initialize database with schema."""
    if not schema_path.exists():
        print(f"ERROR: Schema file not found: {schema_path}")
        return False
    
    # Ensure data directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read schema
    schema_sql = schema_path.read_text()
    
    # Filter out sqlite-vec virtual table if extension not available
    # Check for vec0 module availability
    test_conn = sqlite3.connect(":memory:")
    vec_available = False
    try:
        # Try loading sqlite-vec extension
        test_conn.enable_load_extension(True)
        # Try common extension paths
        for ext_path in ['/usr/lib/sqlite-vec.so', '/usr/local/lib/sqlite-vec.so']:
            if Path(ext_path).exists():
                test_conn.load_extension(ext_path)
                vec_available = True
                break
    except:
        pass
    test_conn.close()
    
    # Remove vec0 virtual table creation if not available
    if not vec_available:
        import re
        # Remove only the CREATE VIRTUAL TABLE ... vec0 block
        schema_sql = re.sub(
            r'CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings USING vec0\([^)]+\);',
            '-- chunk_embeddings virtual table skipped (sqlite-vec not available)',
            schema_sql,
            flags=re.MULTILINE
        )
        print("Note: sqlite-vec extension not available, using fallback embedding storage")
    
    # Connect and execute
    conn = sqlite3.connect(db_path)
    
    try:
        # Execute schema
        conn.executescript(schema_sql)
        
        # Add fallback embedding table if vec not available
        if not vec_available:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS chunk_embeddings_fallback (
                    chunk_id INTEGER PRIMARY KEY,
                    embedding_model TEXT,
                    embedding_data BLOB,  -- Serialized embedding vector
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chunk_id) REFERENCES content_chunks(id)
                )"""
            )
            print("  Created fallback embedding table")
        
        # Verify tables
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        
        print(f"Database initialized: {db_path}")
        print(f"Tables created: {len(table_names)}")
        print(f"  - {', '.join(sorted(table_names))}")
        
        # Check for sqlite-vec virtual table
        vec_tables = [t for t in table_names if 'vec' in t.lower()]
        if vec_tables:
            print(f"Vector tables: {vec_tables}")
        else:
            print("Note: chunk_embeddings virtual table may require sqlite-vec extension")
        
        conn.commit()
        return True
        
    except sqlite3.Error as e:
        print(f"ERROR: Database initialization failed: {e}")
        return False
    finally:
        conn.close()


def verify_database(db_path: Path = DB_PATH) -> bool:
    """Verify database is properly initialized."""
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return False
    
    conn = sqlite3.connect(db_path)
    
    try:
        # Check required tables
        required_tables = [
            'source_files', 'content_records', 'content_chunks',
            'briefing_history', 'briefing_items', 'arxiv_papers',
            'portfolio_drafts', 'approval_events', 'ingestion_batches',
            'workflow_state', 'schema_version'
        ]
        
        existing = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        existing_names = set(t[0] for t in existing)
        
        missing = set(required_tables) - existing_names
        if missing:
            print(f"Missing tables: {missing}")
            return False
        
        # Check schema version
        version = conn.execute(
            "SELECT version FROM schema_version LIMIT 1"
        ).fetchone()
        
        print(f"Database verified: schema version {version[0] if version else 'unknown'}")
        return True
        
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize personal AI database")
    parser.add_argument("--verify", action="store_true", help="Verify existing database")
    parser.add_argument("--db-path", type=Path, default=DB_PATH, help="Database path")
    parser.add_argument("--schema-path", type=Path, default=SCHEMA_PATH, help="Schema path")
    
    args = parser.parse_args()
    
    if args.verify:
        success = verify_database(args.db_path)
    else:
        success = init_database(args.db_path, args.schema_path)
    
    sys.exit(0 if success else 1)
