#!/usr/bin/env python3
"""
Classify source files and determine processing strategy.
Analyzes file types and determines appropriate extraction methods.
"""

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "state.db"


def classify_file(path: Path) -> dict[str, Any]:
    """Classify a source file and determine processing info."""
    suffix = path.suffix.lower()
    name = path.name.lower()
    
    classification = {
        'path': str(path),
        'suffix': suffix,
        'file_type': 'unknown',
        'extractor': 'none',
        'processable': False,
        'estimated_records': 0,
        'notes': []
    }
    
    # JSONL files (conversations)
    if suffix == '.jsonl' or path.suffix == '.jsonl':
        classification['file_type'] = 'jsonl'
        classification['extractor'] = 'jsonl_extractor'
        classification['processable'] = True
        
        # Estimate record count by sampling first lines
        try:
            with open(path) as f:
                line_count = 0
                for _ in range(100):
                    line = f.readline()
                    if not line:
                        break
                    line_count += 1
                # Extrapolate based on file size
                avg_line_size = path.stat().st_size / max(line_count, 1)
                classification['estimated_records'] = int(path.stat().st_size / avg_line_size)
        except Exception as e:
            classification['notes'].append(f"Error estimating records: {e}")
    
    # ZIP archives (Google Takeout, etc.)
    elif suffix == '.zip':
        classification['file_type'] = 'archive'
        classification['extractor'] = 'archive_extractor'
        classification['processable'] = True
        classification['notes'].append("Requires extraction before processing")
        
        # Try to identify archive type by name
        if 'takeout' in name:
            classification['archive_type'] = 'google_takeout'
            classification['extractor'] = 'takeout_extractor'
    
    # JSON files
    elif suffix == '.json':
        classification['file_type'] = 'json'
        classification['extractor'] = 'json_extractor'
        classification['processable'] = True
    
    # Text/markdown files
    elif suffix in ['.txt', '.md', '.markdown']:
        classification['file_type'] = 'text'
        classification['extractor'] = 'text_extractor'
        classification['processable'] = True
    
    # Patch files
    elif suffix == '.patch':
        classification['file_type'] = 'patch'
        classification['extractor'] = 'text_extractor'
        classification['processable'] = True
        classification['notes'].append("Patch file - may contain code diffs")
    
    # Log files
    elif suffix == '.log':
        classification['file_type'] = 'log'
        classification['extractor'] = 'text_extractor'
        classification['processable'] = False  # usually not useful for knowledge
        classification['notes'].append("Log file - typically not indexed")
    
    else:
        classification['notes'].append(f"Unknown file type: {suffix}")
    
    return classification


def run_classification(db_path: Path = DB_PATH) -> dict:
    """Classify all pending source files."""
    
    conn = sqlite3.connect(db_path)
    
    try:
        stats = {
            'started_at': datetime.now().isoformat(),
            'files_classified': 0,
            'files_processable': 0,
            'files_unprocessable': 0,
            'extractors_needed': set(),
            'estimated_total_records': 0
        }
        
        # Get pending files
        pending_files = conn.execute(
            "SELECT id, path, source_type, file_size_bytes FROM source_files "
            "WHERE ingestion_status IN ('pending', 'failed')"
        ).fetchall()
        
        for file_id, path, source_type, file_size in pending_files:
            file_path = Path(path)
            
            if not file_path.exists():
                conn.execute(
                    "UPDATE source_files SET ingestion_status = 'failed', "
                    "error_message = 'File not found' WHERE id = ?",
                    (file_id,)
                )
                continue
            
            classification = classify_file(file_path)
            
            # Update metadata with classification
            existing_meta = conn.execute(
                "SELECT metadata FROM source_files WHERE id = ?",
                (file_id,)
            ).fetchone()
            
            metadata = json.loads(existing_meta[0]) if existing_meta and existing_meta[0] else {}
            metadata['classification'] = classification
            
            conn.execute(
                "UPDATE source_files SET metadata = ? WHERE id = ?",
                (json.dumps(metadata), file_id)
            )
            
            stats['files_classified'] += 1
            if classification['processable']:
                stats['files_processable'] += 1
                stats['extractors_needed'].add(classification['extractor'])
                stats['estimated_total_records'] += classification['estimated_records']
            else:
                stats['files_unprocessable'] += 1
            
            print(f"Classified: {file_path.name} -> {classification['file_type']} "
                  f"(processable: {classification['processable']})")
        
        stats['extractors_needed'] = list(stats['extractors_needed'])
        stats['completed_at'] = datetime.now().isoformat()
        
        conn.commit()
        
        print(f"\nClassification complete:")
        print(f"  Files classified: {stats['files_classified']}")
        print(f"  Processable: {stats['files_processable']}")
        print(f"  Unprocessable: {stats['files_unprocessable']}")
        print(f"  Extractors needed: {stats['extractors_needed']}")
        print(f"  Estimated records: {stats['estimated_total_records']}")
        
        return stats
        
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify source files")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Database path")
    
    args = parser.parse_args()
    
    result = run_classification(args.db)
    sys.exit(0)
