#!/usr/bin/env python3
"""
Discover source files in exports directory and register in manifest.
Scans the configured exports directory for files matching patterns.
"""

import hashlib
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from core.project_paths import EXPORTS_ROOT, STATE_DB_PATH, resolve_project_path

CONFIG_PATH = PROJECT_ROOT / "config" / "sources.yaml"
DB_PATH = STATE_DB_PATH
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load YAML config file."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def compute_file_hash(path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def discover_files(exports_root: Path, patterns: list[str]) -> list[Path]:
    """Discover files matching patterns in exports directory."""
    matches = []
    for pattern in patterns:
        # Use glob for pattern matching
        for match in exports_root.glob(pattern):
            if match.is_file():
                matches.append(match)
        # Also check nested patterns
        for match in exports_root.rglob(pattern.replace('**/', '')):
            if match.is_file():
                matches.append(match)
    return sorted(set(matches))


def register_file(conn: sqlite3.Connection, path: Path, source_type: str, 
                  priority: int, categories: list[str], metadata: dict = None) -> int:
    """Register a source file in the database."""
    file_hash = compute_file_hash(path)
    file_size = path.stat().st_size
    
    # Check if already registered
    existing = conn.execute(
        "SELECT id, file_hash FROM source_files WHERE path = ?",
        (str(path),)
    ).fetchone()
    
    if existing:
        # Update if hash changed
        if existing[1] != file_hash:
            conn.execute(
                """UPDATE source_files 
                   SET file_hash = ?, file_size_bytes = ?, ingestion_status = 'pending',
                       last_ingested_at = NULL, error_message = NULL
                   WHERE id = ?""",
                (file_hash, file_size, existing[0])
            )
            return existing[0]
        return existing[0]  # unchanged
    
    # Insert new file
    result = conn.execute(
        """INSERT INTO source_files 
           (path, source_type, file_hash, file_size_bytes, metadata)
           VALUES (?, ?, ?, ?, ?)""",
        (str(path), source_type, file_hash, file_size, 
         json.dumps(metadata or {'priority': priority, 'categories': categories}))
    )
    return result.lastrowid


def run_discovery(exports_root: Path = None, config_path: Path = CONFIG_PATH,
                  db_path: Path = DB_PATH) -> dict:
    """Run file discovery and registration."""
    
    # Load config
    config = load_yaml_config(config_path)
    
    if exports_root is None:
        exports_root = resolve_project_path(config.get('exports_root', str(EXPORTS_ROOT)))
    
    if not exports_root.exists():
        print(f"ERROR: Exports root not found: {exports_root}")
        return {'error': 'exports_root_not_found', 'files_registered': 0}
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    try:
        stats = {
            'started_at': datetime.now().isoformat(),
            'exports_root': str(exports_root),
            'sources': {},
            'files_registered': 0,
            'files_updated': 0,
            'files_skipped': 0,
            'total_size_bytes': 0
        }
        
        sources = config.get('sources', {})
        
        for source_name, source_config in sources.items():
            pattern = source_config.get('path_pattern', '')
            source_type = source_config.get('type', 'unknown')
            priority = source_config.get('priority', 5)
            categories = source_config.get('categories', [])
            
            print(f"Scanning {source_name} ({pattern})...")
            
            # Discover files
            files = discover_files(exports_root, [pattern])
            
            source_stats = {
                'pattern': pattern,
                'files_found': len(files),
                'files_registered': 0
            }
            
            for file_path in files:
                file_id = register_file(
                    conn, file_path, source_name,
                    priority, categories,
                    {'priority': priority, 'categories': categories}
                )
                
                file_size = file_path.stat().st_size
                stats['total_size_bytes'] += file_size
                source_stats['files_registered'] += 1
                stats['files_registered'] += 1
                
                print(f"  Registered: {file_path.name} ({file_size / 1024 / 1024:.1f} MB)")
            
            stats['sources'][source_name] = source_stats
        
        # Create manifest file
        manifest_path = MANIFEST_DIR / f"discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(stats, indent=2))
        
        stats['manifest_path'] = str(manifest_path)
        stats['completed_at'] = datetime.now().isoformat()
        
        conn.commit()
        
        print(f"\nDiscovery complete:")
        print(f"  Files registered: {stats['files_registered']}")
        print(f"  Total size: {stats['total_size_bytes'] / 1024 / 1024 / 1024:.2f} GB")
        print(f"  Manifest: {manifest_path}")
        
        return stats
        
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Discover source files in exports")
    parser.add_argument("--exports-root", type=Path, help="Exports root directory")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Config path")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Database path")
    
    args = parser.parse_args()
    
    result = run_discovery(args.exports_root, args.config, args.db)
    
    if 'error' in result:
        sys.exit(1)
