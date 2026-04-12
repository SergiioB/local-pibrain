#!/usr/bin/env python3
"""
Chunk content records into smaller pieces for retrieval and embedding.
Uses configurable chunk size with overlap. Stores chunks for embedding.
"""

import hashlib
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "state.db"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
               overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[dict]:
    """Split text into chunks with overlap."""
    
    if not text or len(text.strip()) == 0:
        return []
    
    if len(text) <= chunk_size:
        return [{
            'text': text.strip(),
            'start': 0,
            'end': len(text),
            'index': 0,
            'hash': hashlib.sha256(text.encode()).hexdigest()
        }]
    
    chunks = []
    start = 0
    index = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # Try to find a natural break point
        chunk = text[start:end]
        
        # Prefer breaking at newline or sentence
        break_point = -1
        for bp in ['\n\n', '\n', '. ', '。', '。 ']:
            pos = chunk.rfind(bp)
            if pos > chunk_size // 2:  # Only if reasonable break point
                break_point = pos + len(bp)
                break
        
        if break_point > 0 and end < len(text):
            end = start + break_point
            chunk = text[start:end]
        
        chunk_text_clean = chunk.strip()
        if chunk_text_clean:
            chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()
            
            chunks.append({
                'text': chunk_text_clean,
                'start': start,
                'end': end,
                'index': index,
                'hash': chunk_hash
            })
        
        # Move to next chunk with overlap
        start = end - overlap if end < len(text) else end
        index += 1
    
    return chunks


def get_record_content(conn: sqlite3.Connection, record_id: int) -> str:
    """Extract full content from a record including metadata."""
    
    record = conn.execute(
        """SELECT cr.title, cr.category, cr.metadata, cr.record_type, cr.external_id,
                  sf.path, sf.source_type
           FROM content_records cr
           JOIN source_files sf ON cr.source_file_id = sf.id
           WHERE cr.id = ?""",
        (record_id,)
    ).fetchone()
    
    if not record:
        return ""
    
    title, category, metadata_json, record_type, external_id, source_path, source_type = record
    metadata = json.loads(metadata_json) if metadata_json else {}
    
    content_parts = []
    
    # Add title
    if title:
        content_parts.append(f"Title: {title}")
    
    # Add category
    if category:
        content_parts.append(f"Category: {category}")
    
    # Add metadata content
    if metadata:
        if isinstance(metadata.get('content_text'), str) and len(metadata['content_text']) > 20:
            return metadata['content_text']

        # Extract any content from metadata
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
                # Find the specific record in the JSONL file
                with open(source_file, 'r', encoding='utf-8', errors='replace') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            line_data = json.loads(line.strip())
                            line_id = line_data.get('id') or line_data.get('conversation_id') or f"line_{line_num}"
                            if str(line_id) == str(external_id):
                                # Found the matching record
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
    
    # Fallback: use any stored content representation
    if metadata and 'content_preview' in metadata:
        content_parts.append(metadata['content_preview'])
    
    return '\n\n'.join(content_parts)


def run_chunking(db_path: Path = DB_PATH, chunk_size: int = DEFAULT_CHUNK_SIZE,
                 overlap: int = DEFAULT_CHUNK_OVERLAP) -> dict:
    """Chunk all content records that haven't been chunked yet."""
    
    conn = sqlite3.connect(db_path)
    
    try:
        stats = {
            'started_at': datetime.now().isoformat(),
            'records_chunked': 0,
            'chunks_created': 0,
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
        
        print(f"Found {len(records)} records to chunk")
        
        for i, (record_id, source_file_id) in enumerate(records):
            try:
                # Get the full content for this record
                content_text = get_record_content(conn, record_id)
                
                if not content_text or len(content_text.strip()) < 50:
                    print(f"  Skipping record {record_id}: insufficient content")
                    continue
                
                # Chunk the text
                chunks = chunk_text(content_text, chunk_size, overlap)
                
                if not chunks:
                    print(f"  Skipping record {record_id}: no chunks generated")
                    continue
                
                for chunk_data in chunks:
                    if chunk_data['text'] and len(chunk_data['text']) > 20:
                        conn.execute(
                            """INSERT INTO content_chunks 
                               (content_record_id, chunk_index, chunk_text, chunk_hash,
                                start_char, end_char, embedding_status, metadata)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                            (record_id, chunk_data['index'], chunk_data['text'],
                             chunk_data['hash'], chunk_data['start'], chunk_data['end'],
                             'pending', json.dumps({}))
                        )
                        stats['chunks_created'] += 1
                
                stats['records_chunked'] += 1
                
                if stats['records_chunked'] % 50 == 0:
                    conn.commit()
                    print(f"  Chunked {stats['records_chunked']}/{len(records)} records... ({stats['chunks_created']} chunks)")
                
            except Exception as e:
                stats['errors'].append({'record_id': record_id, 'error': str(e)})
                print(f"  ERROR on record {record_id}: {e}")
        
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
        print(f"  Errors: {len(stats['errors'])}")
        
        return stats
        
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chunk content records")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Database path")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    
    args = parser.parse_args()
    
    result = run_chunking(args.db, args.chunk_size, args.overlap)
    sys.exit(0)
