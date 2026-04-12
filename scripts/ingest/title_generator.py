#!/usr/bin/env python3
"""
Generate better titles for content records using local llama.cpp inference.
Integrates with local models for title extraction and summarization.
"""

import json
import sqlite3
import sys
import subprocess
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "state.db"
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from core.project_paths import MODELS_DIR

# Default small model for title extraction
DEFAULT_MODEL = "qwen3.5-4b-q4_k_m.gguf"


def find_llama_cpp() -> Optional[Path]:
    """Find llama-cli or llama-server binary."""
    # Check common locations
    paths = [
        PROJECT_ROOT / "third_party" / "llama.cpp" / "build" / "bin" / "llama-cli",
        PROJECT_ROOT / "third_party" / "rk-llama.cpp" / "build" / "bin" / "llama-cli",
        Path("/usr/local/bin/llama-cli"),
        Path.home() / "llama.cpp/build/bin/llama-cli",
    ]
    
    for p in paths:
        if p.exists():
            return p
    
    # Try which command
    try:
        result = subprocess.run(
            ["which", "llama-cli"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except:
        pass
    
    return None


def find_model() -> Optional[Path]:
    """Find a suitable model for title generation."""
    # Try default first
    model_path = MODELS_DIR / DEFAULT_MODEL
    if model_path.exists():
        return model_path
    
    # Find any GGUF file
    if MODELS_DIR.exists():
        for gguf in MODELS_DIR.glob("*.gguf"):
            return gguf
    
    return None


def generate_title_with_llama(content: str, llama_path: Path, model_path: Path) -> Optional[str]:
    """Generate a title using local llama.cpp model."""
    
    # Truncate content for prompt
    content_preview = content[:1000].replace('"', '"').replace('\n', ' ')
    
    prompt = f"""<|im_start|>system
You are a helpful assistant that generates concise, descriptive titles for conversations and documents. Generate a title of 5-10 words that captures the main topic.<|im_end|>
<|im_start|>user
Generate a title for this content:

{content_preview}...

Title:<|im_end|>
<|im_start|>assistant
"""
    
    try:
        result = subprocess.run(
            [
                str(llama_path),
                "-m", str(model_path),
                "-p", prompt,
                "-n", "50",
                "--temp", "0.3",
                "--top-p", "0.9",
                "--no-display-prompt"
            ],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            title = result.stdout.strip()
            # Clean up title
            title = title.split('\n')[0].strip()
            if title and len(title) > 5:
                return title[:100]  # Limit length
        
        return None
        
    except subprocess.TimeoutExpired:
        print("    Timeout generating title")
        return None
    except Exception as e:
        print(f"    Error generating title: {e}")
        return None


def extract_title_heuristic(content: str, category: str, metadata: dict) -> Optional[str]:
    """Extract title using heuristics when model is unavailable."""
    
    # Check metadata first
    if metadata:
        for key in ['title', 'name', 'subject', 'topic']:
            if key in metadata and metadata[key]:
                val = metadata[key]
                if isinstance(val, str) and len(val) > 3 and len(val) < 100:
                    return val
    
    # Extract from content
    lines = content.split('\n')
    
    # Look for explicit title markers
    for line in lines[:10]:
        line = line.strip()
        if line.startswith('Title:') or line.startswith('# '):
            title = line.split(':', 1)[-1].split('#', 1)[-1].strip()
            if len(title) > 5 and len(title) < 100:
                return title
    
    # For code content, look for function/class names
    if category == 'code':
        for line in lines[:30]:
            if 'def ' in line or 'class ' in line:
                parts = line.split('def ')[-1].split('class ')[-1]
                name = parts.split('(')[0].split(':')[0].strip()
                if name and name[0].isalpha():
                    return f"Code: {name}"
    
    # Extract first substantial sentence
    for line in lines[:5]:
        line = line.strip()
        if len(line) > 20 and len(line) < 100:
            # Remove common prefixes
            prefixes = ['User:', 'Assistant:', 'System:', 'Human:', 'AI:']
            for prefix in prefixes:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
            if len(line) > 20:
                return line[:100]
    
    # Generate from category + first words
    words = content.split()[:15]
    if words:
        snippet = ' '.join(words)
        return f"[{category.upper()}] {snippet[:80]}..."
    
    return None


def get_record_full_content(conn: sqlite3.Connection, record_id: int) -> str:
    """Get full content text for a record."""
    
    # Get record info
    record = conn.execute(
        """SELECT cr.title, cr.category, cr.metadata, cr.external_id,
                  cr.record_type, sf.path, sf.source_type
           FROM content_records cr
           JOIN source_files sf ON cr.source_file_id = sf.id
           WHERE cr.id = ?""",
        (record_id,)
    ).fetchone()
    
    if not record:
        return ""
    
    title, category, metadata_json, external_id, record_type, source_path, source_type = record
    metadata = json.loads(metadata_json) if metadata_json else {}
    
    content_parts = []
    
    # Try to read from source file
    if source_path and Path(source_path).exists():
        try:
            with open(source_path, 'r', encoding='utf-8', errors='replace') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        line_id = data.get('id') or data.get('conversation_id') or f"line_{line_num}"
                        if str(line_id) == str(external_id):
                            # Extract messages
                            messages = data.get('messages') or data.get('turns') or []
                            for msg in messages[:10]:  # Limit messages
                                role = msg.get('role', 'unknown')
                                content = msg.get('content') or msg.get('text', '')
                                if isinstance(content, list):
                                    content = ' '.join(str(c) for c in content)
                                if content:
                                    content_parts.append(f"{role}: {content}")
                            break
                    except:
                        continue
        except:
            pass
    
    # Use metadata content if available
    if not content_parts and metadata:
        for key in ['content', 'text', 'description', 'full_text']:
            if key in metadata:
                val = metadata[key]
                if isinstance(val, str):
                    content_parts.append(val)
                elif isinstance(val, list):
                    content_parts.extend(str(v) for v in val)
    
    return '\n'.join(content_parts)


def run_title_generation(db_path: Path = DB_PATH, use_llama: bool = True, 
                         batch_size: int = 50) -> dict:
    """Generate titles for records that don't have them."""
    
    conn = sqlite3.connect(db_path)
    
    try:
        stats = {
            'started_at': datetime.now().isoformat(),
            'records_processed': 0,
            'titles_generated': 0,
            'titles_heuristic': 0,
            'errors': [],
            'llama_available': False
        }
        
        # Check llama availability
        llama_path = None
        model_path = None
        
        if use_llama:
            llama_path = find_llama_cpp()
            model_path = find_model()
            if llama_path and model_path:
                stats['llama_available'] = True
                print(f"Using llama.cpp: {llama_path}")
                print(f"Using model: {model_path.name}")
            else:
                if not llama_path:
                    print("llama-cli not found, using heuristic mode")
                if not model_path:
                    print("No GGUF model found, using heuristic mode")
        
        # Get records without good titles
        records = conn.execute(
            """SELECT cr.id, cr.title, cr.category, cr.metadata, cr.external_id
               FROM content_records cr
               WHERE cr.title IS NULL 
                  OR cr.title = '' 
                  OR cr.title = 'Untitled'
                  OR cr.title LIKE '%Untitled%'
               LIMIT ?""",
            (batch_size * 2,)
        ).fetchall()
        
        print(f"Found {len(records)} records needing titles")
        
        for record_id, current_title, category, metadata_json, external_id in records:
            try:
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                # Get content for title generation
                content = get_record_full_content(conn, record_id)
                
                if not content:
                    print(f"  Skipping record {record_id}: no content")
                    continue
                
                # Try llama first, then heuristic
                new_title = None
                
                if stats['llama_available'] and len(content) > 100:
                    print(f"  Generating title for record {record_id}...")
                    new_title = generate_title_with_llama(content, llama_path, model_path)
                    if new_title:
                        stats['titles_generated'] += 1
                
                if not new_title:
                    new_title = extract_title_heuristic(content, category, metadata)
                    if new_title:
                        stats['titles_heuristic'] += 1
                
                if new_title:
                    conn.execute(
                        "UPDATE content_records SET title = ? WHERE id = ?",
                        (new_title, record_id)
                    )
                    print(f"  -> {new_title[:60]}...")
                else:
                    print(f"  Could not generate title for record {record_id}")
                
                stats['records_processed'] += 1
                
                if stats['records_processed'] % 10 == 0:
                    conn.commit()
                    
            except Exception as e:
                stats['errors'].append({'record_id': record_id, 'error': str(e)})
                print(f"  ERROR on record {record_id}: {e}")
        
        conn.commit()
        stats['completed_at'] = datetime.now().isoformat()
        
        print(f"\nTitle generation complete:")
        print(f"  Records processed: {stats['records_processed']}")
        print(f"  Titles with llama: {stats['titles_generated']}")
        print(f"  Titles heuristic: {stats['titles_heuristic']}")
        print(f"  Errors: {len(stats['errors'])}")
        
        return stats
        
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate titles for content records")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Database path")
    parser.add_argument("--no-llama", action="store_true", help="Skip llama.cpp, use heuristics only")
    parser.add_argument("--batch-size", type=int, default=50)
    
    args = parser.parse_args()
    
    result = run_title_generation(args.db, not args.no_llama, args.batch_size)
    sys.exit(0)
