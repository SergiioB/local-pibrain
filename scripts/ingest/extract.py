#!/usr/bin/env python3
"""
Extract text content from source files.
Processes JSONL conversation files, JSON exports, and text files.
"""

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from takeout_extractor import extract_from_takeout_zip

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "state.db"


def parse_jsonl_line(line: str) -> dict[str, Any] | None:
    """Parse a JSONL line, handling malformed data."""
    try:
        return json.loads(line.strip())
    except json.JSONDecodeError:
        return None


def coerce_text(value: Any) -> str:
    """Flatten provider-specific message content into plain text."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [coerce_text(item) for item in value]
        return '\n'.join(part for part in parts if part).strip()
    if isinstance(value, dict):
        if isinstance(value.get('text'), str):
            return value['text']
        if isinstance(value.get('description'), str) and isinstance(value.get('subject'), str):
            return ""
        if isinstance(value.get('content'), str):
            return value['content']
    return ""


def clean_message_text(text: str) -> str:
    """Drop thinking artifacts and serialized reasoning traces."""
    if not text:
        return ""

    without_thinking = text.replace("<thinking>", "\n").replace("</thinking>", "\n")
    cleaned_blocks = []
    for block in without_thinking.split('\n\n'):
        block = block.strip()
        if not block:
            continue
        if looks_like_thought_artifact(block):
            continue
        cleaned_blocks.append(block)
    return '\n\n'.join(cleaned_blocks).strip()


def looks_like_thought_artifact(block: str) -> bool:
    return (
        (block.startswith('{') or block.startswith('['))
        and '"description"' in block
        and '"subject"' in block
    )


def timestamp_to_iso(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000).isoformat()
    return None


def extract_jsonl_conversations(path: Path) -> Iterator[dict[str, Any]]:
    """Extract conversation records from JSONL file."""
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line_num, line in enumerate(f, 1):
            record = parse_jsonl_line(line)
            if not record:
                continue
            
            # Normalize conversation structure
            # Different sources have different formats
            conv = {
                'external_id': record.get('id') or record.get('conversation_id') or record.get('session_id') or f"line_{line_num}",
                'record_type': 'conversation',
                'title': record.get('title') or record.get('name') or record.get('session_title') or record.get('session_display_title') or None,
                'created_at': None,
                'updated_at': None,
                'category': None,
                'tags': [],
                'content': [],
                'metadata': {
                    'source': record.get('source'),
                    'project_path': record.get('project_path') or record.get('cwd'),
                    'source_file': record.get('source_file') or record.get('session_file'),
                }
            }
            
            # Parse timestamps
            for ts_field in ['created_at', 'createdAt', 'timestamp', 'date']:
                if record.get(ts_field):
                    conv['created_at'] = timestamp_to_iso(record.get(ts_field))
                    break
            for ts_field in ['updated_at', 'updatedAt', 'last_updated']:
                if record.get(ts_field):
                    conv['updated_at'] = timestamp_to_iso(record.get(ts_field))
                    break
            
            # Parse messages
            messages = record.get('messages') or record.get('turns') or []
            for msg in messages:
                raw_content = msg.get('content') if isinstance(msg, dict) else ''
                if not raw_content and isinstance(msg, dict):
                    raw_content = msg.get('text') or msg.get('message') or ''
                content_text = clean_message_text(coerce_text(raw_content))
                msg_content = {
                    'role': msg.get('role') or msg.get('sender') or 'unknown',
                    'content': content_text,
                    'timestamp': timestamp_to_iso(msg.get('timestamp') or msg.get('createdAt'))
                }
                if msg_content['content']:
                    conv['content'].append(msg_content)
            
            # Determine category from content
            if conv['content']:
                # Handle content that might be string or list
                content_parts = []
                for m in conv['content']:
                    c = m.get('content', '')
                    if isinstance(c, list):
                        content_parts.extend(str(item) for item in c)
                    elif isinstance(c, str):
                        content_parts.append(c)
                total_content = ' '.join(content_parts)
                if any(kw in total_content.lower() for kw in ['def ', 'class ', 'function ', 'import ', 'code']):
                    conv['category'] = 'code'
                elif any(kw in total_content.lower() for kw in ['tool', 'execute', 'script', 'command']):
                    conv['category'] = 'tool'
                elif any(kw in total_content.lower() for kw in ['think', 'reason', 'analysis', 'consider']):
                    conv['category'] = 'reasoning'
                else:
                    conv['category'] = 'chat'
            
            if conv['content']:
                yield conv


def extract_normalized_conversations(path: Path) -> Iterator[dict[str, Any]]:
    """Extract from normalized conversation format (reap output)."""
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line_num, line in enumerate(f, 1):
            record = parse_jsonl_line(line)
            if not record:
                continue
            
            # normalized_conversations.jsonl format from qwen35-a3b-reap
            conv = {
                'external_id': record.get('conversation_id') or record.get('session_id') or record.get('id') or f"norm_{line_num}",
                'record_type': 'conversation',
                'title': record.get('title') or record.get('name'),
                'created_at': timestamp_to_iso(record.get('timestamp') or record.get('created_at')),
                'updated_at': timestamp_to_iso(record.get('updated_at') or record.get('last_updated')),
                'category': record.get('category', 'chat'),
                'tags': record.get('tags', []),
                'content': [],
                'metadata': {
                    'source': record.get('source'),
                    'session_id': record.get('session_id'),
                    'conversation_id': record.get('conversation_id'),
                }
            }

            messages = record.get('messages') or []
            for msg in messages:
                text = clean_message_text(coerce_text(msg.get('text') if isinstance(msg, dict) else ''))
                if not text:
                    continue
                conv['content'].append({
                    'role': msg.get('role', 'unknown'),
                    'content': text,
                    'timestamp': timestamp_to_iso(msg.get('timestamp')) if isinstance(msg, dict) else None,
                })

            if conv['content']:
                total_content = ' '.join(str(m.get('content', '')) for m in conv['content'])
                if not conv['category']:
                    if any(kw in total_content.lower() for kw in ['def ', 'class ', 'function ', 'import ', 'code']):
                        conv['category'] = 'code'
                    elif any(kw in total_content.lower() for kw in ['tool', 'execute', 'script', 'command']):
                        conv['category'] = 'tool'
                    elif any(kw in total_content.lower() for kw in ['analysis', 'consider', 'reason']):
                        conv['category'] = 'reasoning'
                    else:
                        conv['category'] = 'chat'
                yield conv


def extract_text_file(path: Path) -> Iterator[dict[str, Any]]:
    """Extract from plain text/markdown file."""
    content = path.read_text(encoding='utf-8', errors='replace')
    
    yield {
        'external_id': path.stem,
        'record_type': 'document',
        'title': path.name,
        'created_at': datetime.now().isoformat(),
        'category': 'note',
        'tags': [],
        'content': [{'role': 'document', 'content': content}],
        'metadata': {'file_type': path.suffix}
    }


def insert_content_record(conn: sqlite3.Connection, source_file_id: int,
                          record: dict[str, Any]) -> int:
    """Insert a content record into database."""
    
    # Calculate importance score based on content
    # Handle content that might be string or list
    content_parts = []
    for m in record.get('content', []):
        c = m.get('content', '')
        if isinstance(c, list):
            content_parts.extend(str(item) for item in c)
        elif isinstance(c, str):
            content_parts.append(c)
    content_text = ' '.join(content_parts)
    metadata = dict(record.get('metadata', {}))
    if content_text:
        metadata.setdefault('content_preview', content_text[:2000])
        metadata.setdefault('content_text', content_text)
    importance = 0.5
    
    # Boost importance for code/tool content
    if record.get('category') == 'code':
        importance = 0.7
    elif record.get('category') == 'tool':
        importance = 0.6
    elif record.get('category') == 'reasoning':
        importance = 0.65
    
    # Boost for longer content
    if len(content_text) > 1000:
        importance += 0.1
    
    result = conn.execute(
        """INSERT INTO content_records 
           (source_file_id, record_type, external_id, title, created_at, 
            updated_at, category, tags, importance_score, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (source_file_id, record['record_type'], record['external_id'],
         record.get('title'), record.get('created_at'), record.get('updated_at'),
         record.get('category'), json.dumps(record.get('tags', [])), importance,
         json.dumps(metadata))
    )
    
    return result.lastrowid


def run_extraction(db_path: Path = DB_PATH, batch_size: int = 1000) -> dict:
    """Run content extraction for processable files."""
    
    conn = sqlite3.connect(db_path)
    
    try:
        stats = {
            'started_at': datetime.now().isoformat(),
            'files_processed': 0,
            'records_created': 0,
            'errors': [],
            'files_skipped': 0
        }
        
        # Create ingestion batch
        batch_result = conn.execute(
            """INSERT INTO ingestion_batches (batch_type, status)
               VALUES ('incremental', 'running')"""
        )
        batch_id = batch_result.lastrowid
        
        # Get processable files
        processable = conn.execute(
            """SELECT id, path, source_type, metadata FROM source_files 
               WHERE ingestion_status IN ('pending', 'failed')"""
        ).fetchall()
        
        for file_id, path, source_type, metadata_json in processable:
            metadata = json.loads(metadata_json) if metadata_json else {}
            classification = metadata.get('classification', {})
            
            if not classification.get('processable'):
                stats['files_skipped'] += 1
                continue
            
            file_path = Path(path)
            extractor = classification.get('extractor', 'none')
            
            print(f"Extracting: {file_path.name} (using {extractor})")
            
            try:
                records_in_file = 0
                
                # Choose extractor based on classification
                if extractor == 'jsonl_extractor':
                    if 'normalized' in file_path.name or 'reap' in str(file_path):
                        records_iter = extract_normalized_conversations(file_path)
                    else:
                        records_iter = extract_jsonl_conversations(file_path)
                elif extractor == 'takeout_extractor':
                    records_iter = extract_from_takeout_zip(file_path)
                elif extractor == 'text_extractor':
                    records_iter = extract_text_file(file_path)
                else:
                    print(f"  No extractor for: {extractor}")
                    stats['files_skipped'] += 1
                    continue
                
                for record in records_iter:
                    content_id = insert_content_record(conn, file_id, record)
                    records_in_file += 1
                    stats['records_created'] += 1
                    
                    if records_in_file % batch_size == 0:
                        conn.commit()
                        print(f"  Extracted {records_in_file} records...")
                
                # Update source file status
                conn.execute(
                    """UPDATE source_files 
                       SET ingestion_status = 'ingested', 
                           last_ingested_at = ?, 
                           record_count = ?
                       WHERE id = ?""",
                    (datetime.now().isoformat(), records_in_file, file_id)
                )
                
                stats['files_processed'] += 1
                print(f"  Complete: {records_in_file} records")
                
            except Exception as e:
                error_msg = str(e)
                stats['errors'].append({'file': path, 'error': error_msg})
                
                conn.execute(
                    """UPDATE source_files 
                       SET ingestion_status = 'failed', error_message = ?
                       WHERE id = ?""",
                    (error_msg, file_id)
                )
                print(f"  ERROR: {error_msg}")
        
        # Update batch status
        conn.execute(
            """UPDATE ingestion_batches 
               SET status = 'completed', completed_at = ?, 
                   files_processed = ?, records_created = ?
               WHERE id = ?""",
            (datetime.now().isoformat(), stats['files_processed'],
             stats['records_created'], batch_id)
        )
        
        # Update workflow state
        conn.execute(
            """UPDATE workflow_state 
               SET value = ?, updated_at = ?
               WHERE key = 'last_ingestion'""",
            (json.dumps({
                'timestamp': datetime.now().isoformat(),
                'batch_id': batch_id,
                'records_created': stats['records_created']
            }), datetime.now().isoformat())
        )
        
        stats['completed_at'] = datetime.now().isoformat()
        stats['batch_id'] = batch_id
        
        conn.commit()
        
        print(f"\nExtraction complete:")
        print(f"  Files processed: {stats['files_processed']}")
        print(f"  Records created: {stats['records_created']}")
        print(f"  Errors: {len(stats['errors'])}")
        
        return stats
        
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract content from source files")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Database path")
    parser.add_argument("--batch-size", type=int, default=1000, help="Commit batch size")
    
    args = parser.parse_args()
    
    result = run_extraction(args.db, args.batch_size)
    
    if result.get('errors'):
        sys.exit(1)
