#!/usr/bin/env python3
"""
Enrich existing content records with structured metadata.
Adds summary, task_type, tools_mentioned, files_mentioned for better retrieval.
"""
import json
import sqlite3
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

DB_PATH = Path("data/state.db")

# Common patterns to detect from content
TOOL_PATTERNS = {
    "claude": ["claude", "anthropic", "claude code"],
    "codex": ["codex", "openai codex"],
    "pi": ["pi-brain", "pi mono"],
    "opencode": ["opencode", "open code"],
    "cursor": ["cursor"],
    "gemini": ["gemini", "google ai"],
    "llama": ["llama.cpp", "llama-server", "llamacpp"],
    "ollama": ["ollama"],
    "vscode": ["vscode", "vscode_tunnel"],
}

TASK_PATTERNS = {
    "bugfix": ["fix", "bug", "error", "crash", "issue", "problem", "broken"],
    "feature": ["feature", "implement", "add", "new", "create"],
    "refactor": ["refactor", "restructure", "reorganize", "cleanup", "clean up"],
    "config": ["config", "setup", "install", "deploy", "environment"],
    "debug": ["debug", "debugging", "trace", "log", "inspect"],
    "research": ["research", "investigate", "explore", "analyze"],
    "testing": ["test", "unit test", "integration test", "pytest"],
    "doc": ["document", "readme", "docs", "write up"],
}

def detect_tools(text):
    found = []
    lower = text.lower()
    for tool, patterns in TOOL_PATTERNS.items():
        for p in patterns:
            if p in lower:
                found.append(tool)
                break
    return list(set(found))

def detect_task(text):
    found = []
    lower = text.lower()
    for task, patterns in TASK_PATTERNS.items():
        for p in patterns:
            if p in lower:
                found.append(task)
                break
    return list(set(found))

def extract_files(text):
    """Extract file paths from text."""
    files = re.findall(r'[/\\][\w\-\.]+(?:/[\w\-\.]+){1,5}', text)
    # Also find common file patterns
    files += re.findall(r'[\w\-]+\.(?:py|js|ts|json|yaml|yml|md|txt|sh|rs|go|kt)', text)
    return list(set(files))[:10]

def generate_summary(text, max_len=200):
    """Generate a simple summary from first non-code lines."""
    lines = text.split('\n')
    summary_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip code lines
        if line.startswith(('%', '$', '#', '```', 'import ', 'from ', 'const ', 'function ', 'def ', '{', '}')):
            continue
        if line.startswith('user:') or line.startswith('assistant:'):
            line = line[10:].strip()
        if line and len(line) > 20:
            summary_lines.append(line)
        if len('\n'.join(summary_lines)) >= max_len:
            break
    return '\n'.join(summary_lines)[:max_len]

def enrich_record(record_text):
    """Extract structured metadata from a record."""
    return {
        "tools": detect_tools(record_text),
        "task_types": detect_task(record_text),
        "files": extract_files(record_text),
        "summary": generate_summary(record_text),
    }

def run_enrichment():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Get all content_records that don't have enriched metadata
    records = conn.execute(
        """SELECT cr.id, cr.title, cr.metadata, sf.source_type
           FROM content_records cr
           JOIN source_files sf ON cr.source_file_id = sf.id
           LIMIT 500"""
    ).fetchall()

    print(f"Processing {len(records)} records...")
    updated = 0
    t0 = time.perf_counter()

    for rec in records:
        rec_id = rec[0]
        title = rec[1]
        meta_json = rec[2]
        source = rec[3]

        if not meta_json:
            continue

        try:
            meta = json.loads(meta_json)
        except json.JSONDecodeError:
            continue

        # Skip if already enriched
        if "enriched" in meta:
            continue

        # Get content text for analysis
        content_text = meta.get("content_text", meta.get("content_preview", ""))
        if not content_text or len(content_text) < 50:
            # Get from chunks
            chunks = conn.execute(
                "SELECT chunk_text FROM content_chunks WHERE content_record_id = ? LIMIT 5",
                (rec_id,)
            ).fetchall()
            content_text = "\n".join(c[0] for c in chunks if c[0])

        if not content_text or len(content_text) < 50:
            continue

        enrichment = enrich_record(content_text)

        # Add to metadata
        meta["enriched"] = True
        meta["tools_mentioned"] = enrichment["tools"]
        meta["task_types"] = enrichment["task_types"]
        meta["files_mentioned"] = enrichment["files"]
        meta["auto_summary"] = enrichment["summary"]

        conn.execute(
            "UPDATE content_records SET metadata = ? WHERE id = ?",
            (json.dumps(meta), rec_id)
        )
        updated += 1

        if updated % 50 == 0:
            conn.commit()
            elapsed = time.perf_counter() - t0
            print(f"  Enriched {updated}/{len(records)} ({elapsed:.0f}s)")

    conn.commit()
    elapsed = time.perf_counter() - t0
    print(f"\nEnrichment complete:")
    print(f"  Records enriched: {updated}")
    print(f"  Time: {elapsed:.1f}s")

    # Also enrich chunks with searchable keywords
    print("\nEnriching chunks with keyword index...")
    t0 = time.perf_counter()
    chunk_updated = 0

    # Update metadata for chunks to include tokenized keywords
    chunks = conn.execute(
        "SELECT id, chunk_text FROM content_chunks WHERE embedding_status = 'embedded' LIMIT 10000"
    ).fetchall()

    for chunk_id, chunk_text in chunks:
        if not chunk_text:
            continue

        # Extract keywords
        words = re.findall(r'[a-z]{4,}', chunk_text.lower())
        # Remove common words
        stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'they', 'their', 'there', 'would', 'could', 'should'}
        keywords = list(set(w for w in words if w not in stopwords))[:30]

        try:
            meta = json.loads(chunk_text) if chunk_text.startswith('{') else {}
        except:
            meta = {}

        meta["keywords"] = keywords

        # Store in chunk metadata if we add a metadata column
        # For now, just count
        chunk_updated += 1

    print(f"  Keywords extracted for {chunk_updated} chunks")
    print(f"  Time: {time.perf_counter() - t0:.1f}s")

    conn.close()

if __name__ == "__main__":
    run_enrichment()
