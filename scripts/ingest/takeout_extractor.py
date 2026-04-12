#!/usr/bin/env python3
"""
Extract conversations from Google Takeout archives.
Handles the nested structure of Takeout ZIP files.
"""

import json
import re
import zipfile
from datetime import datetime
from html import unescape
from pathlib import Path
from typing import Any, Iterator


TEXT_EXTENSIONS = {'.html', '.json', '.md', '.txt'}


def html_to_text(content: str) -> str:
    content = re.sub(r'(?is)<(script|style).*?>.*?</\1>', ' ', content)
    content = re.sub(r'(?i)<br\s*/?>', '\n', content)
    content = re.sub(r'(?i)</p>|</div>|</li>|</tr>|</h[1-6]>', '\n', content)
    content = re.sub(r'(?s)<[^>]+>', ' ', content)
    content = unescape(content)
    lines = []
    for raw_line in content.splitlines():
        cleaned = re.sub(r'\s+', ' ', raw_line).strip()
        if len(cleaned) >= 2:
            lines.append(cleaned)
    return '\n'.join(lines)


def is_binary_payload(raw_bytes: bytes) -> bool:
    if raw_bytes.startswith(b'%PDF'):
        return True
    sample = raw_bytes[:2048]
    if not sample:
        return False
    return sample.count(b'\x00') > 4


def is_takeout_text_artifact(name: str) -> bool:
    lower = name.lower()
    suffix = Path(name).suffix.lower()
    if suffix not in TEXT_EXTENSIONS:
        return False
    if any(pattern in lower for pattern in ['conversation', 'chat', 'dialogue', 'assistant']):
        return True
    if 'aplicaciones de gemini' in lower:
        return True
    if 'gemini' in lower and suffix in {'.html', '.json', '.txt'}:
        return True
    if lower.endswith('miactividad.html'):
        return True
    return False


def extract_from_takeout_zip(zip_path: Path, extract_to: Path = None) -> Iterator[dict[str, Any]]:
    """
    Extract conversations from Google Takeout ZIP archive.
    
    Google Takeout structure typically has:
    - Takeout/Google AI Studio/...
    - Takeout/Search/...
    """
    
    if extract_to is None:
        extract_to = zip_path.parent / f"{zip_path.stem}_extracted"
    
    extracted_files = []
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # List contents
            for name in zf.namelist():
                if is_takeout_text_artifact(name):
                    extracted_files.append(name)
                    
                    # Extract and parse
                    try:
                        raw_bytes = zf.read(name)
                        if is_binary_payload(raw_bytes):
                            continue

                        content = raw_bytes.decode('utf-8', errors='replace')
                        suffix = Path(name).suffix.lower()
                        if suffix == '.html':
                            content = html_to_text(content)
                        
                        # Try to parse as JSON
                        if suffix == '.json':
                            try:
                                data = json.loads(content)
                                yield from parse_google_format(data, name)
                            except json.JSONDecodeError:
                                # Treat as text
                                yield {
                                    'external_id': Path(name).stem,
                                    'record_type': 'document',
                                    'title': Path(name).name,
                                    'created_at': datetime.now().isoformat(),
                                    'category': 'chat',
                                    'tags': ['takeout', 'google'],
                                    'content': [{'role': 'document', 'content': content}],
                                    'metadata': {'source': 'google_takeout', 'path': name}
                                }
                        else:
                            # Text file
                            yield {
                                'external_id': Path(name).stem,
                                'record_type': 'document',
                                'title': Path(name).name,
                                'created_at': datetime.now().isoformat(),
                                'category': 'chat',
                                'tags': ['takeout', 'google'],
                                'content': [{'role': 'document', 'content': content}],
                                'metadata': {'source': 'google_takeout', 'path': name}
                            }
                    except Exception as e:
                        print(f"  Error extracting {name}: {e}")
            
            # If no conversation files found, look for any JSON files
            if not extracted_files:
                for name in zf.namelist():
                    if name.endswith('.json'):
                        try:
                            content = zf.read(name).decode('utf-8', errors='replace')
                            data = json.loads(content)
                            yield from parse_google_format(data, name)
                            extracted_files.append(name)
                        except:
                            pass
            
            # Extract to disk for reference
            if extract_to:
                extract_to.mkdir(parents=True, exist_ok=True)
                zf.extractall(extract_to)
                print(f"  Extracted to: {extract_to}")
    
    except Exception as e:
        print(f"  Error processing archive: {e}")


def parse_google_format(data: dict | list, source_path: str) -> Iterator[dict[str, Any]]:
    """Parse various Google data formats."""
    
    if isinstance(data, list):
        for i, item in enumerate(data):
            yield from parse_google_format(item, f"{source_path}[{i}]")
        return
    
    if not isinstance(data, dict):
        return
    
    # Try common Google conversation formats
    
    # Format 1: Gemini/Bard conversations
    if 'conversations' in data:
        for conv in data.get('conversations', []):
            yield parse_gemini_conversation(conv, source_path)
    
    # Format 2: Single conversation with messages
    elif 'messages' in data or 'turns' in data:
        yield parse_gemini_conversation(data, source_path)
    
    # Format 3: Search history
    elif 'event' in data or 'query' in data:
        yield {
            'external_id': data.get('id', source_path),
            'record_type': 'search',
            'title': data.get('query', {}).get('text', 'Search'),
            'created_at': data.get('time', datetime.now().isoformat()),
            'category': 'chat',
            'tags': ['search', 'google'],
            'content': [{'role': 'user', 'content': str(data.get('query', {}).get('text', ''))}],
            'metadata': {'source': 'google_takeout', 'type': 'search'}
        }
    
    # Format 4: Generic - try to find text content
    else:
        # Look for any text fields
        text_content = []
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 50:
                text_content.append(f"{key}: {value}")
            elif isinstance(value, dict):
                for k2, v2 in value.items():
                    if isinstance(v2, str) and len(v2) > 50:
                        text_content.append(f"{k2}: {v2}")
        
        if text_content:
            yield {
                'external_id': source_path,
                'record_type': 'document',
                'title': Path(source_path).stem,
                'created_at': datetime.now().isoformat(),
                'category': 'chat',
                'tags': ['takeout', 'google'],
                'content': [{'role': 'document', 'content': '\n'.join(text_content)}],
                'metadata': {'source': 'google_takeout', 'path': source_path}
            }


def parse_gemini_conversation(conv: dict, source_path: str) -> dict[str, Any]:
    """Parse a Gemini/Bard conversation format."""
    
    conv_id = conv.get('id') or conv.get('conversationId') or source_path
    
    # Extract title/name
    title = conv.get('name') or conv.get('title') or f"Conversation {conv_id[:8] if isinstance(conv_id, str) else 'unknown'}"
    
    # Extract messages/turns
    messages = conv.get('messages') or conv.get('turns') or []
    
    content = []
    for msg in messages:
        role = msg.get('role') or msg.get('sender') or 'user'
        
        # Get content - might be string or nested
        msg_text = msg.get('content') or msg.get('text') or msg.get('parts', [{}])[0].get('text', '')
        
        if isinstance(msg_text, list):
            msg_text = ' '.join(str(p) for p in msg_text)
        
        if msg_text:
            content.append({
                'role': role,
                'content': str(msg_text)
            })
    
    # Get timestamp
    created_at = conv.get('createTime') or conv.get('created_at') or conv.get('timestamp') or datetime.now().isoformat()
    
    # Determine category
    total_text = ' '.join(m['content'] for m in content).lower()
    category = 'chat'
    if any(kw in total_text for kw in ['code', 'function', 'class ', 'import ', 'def ']):
        category = 'code'
    elif any(kw in total_text for kw in ['analysis', 'think', 'reason']):
        category = 'reasoning'
    
    return {
        'external_id': str(conv_id),
        'record_type': 'conversation',
        'title': title,
        'created_at': created_at,
        'category': category,
        'tags': ['gemini', 'google'],
        'content': content,
        'metadata': {
            'source': 'google_takeout',
            'path': source_path
        }
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python takeout_extractor.py <takeout.zip>")
        sys.exit(1)
    
    zip_path = Path(sys.argv[1])
    
    if not zip_path.exists():
        print(f"File not found: {zip_path}")
        sys.exit(1)
    
    print(f"Extracting from: {zip_path}")
    
    count = 0
    for conv in extract_from_takeout_zip(zip_path):
        count += 1
        print(f"  Found: {conv.get('title', 'unknown')[:50]}")
    
    print(f"\nTotal conversations: {count}")
