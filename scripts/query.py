#!/usr/bin/env python3
"""
Query the knowledge base for relevant content.
Supports keyword search, category filtering, and recency sorting.
"""

import json
import sqlite3
import sys
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib import error, request

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "state.db"

STOP_WORDS = {
    'a', 'about', 'all', 'an', 'and', 'are', 'as', 'at', 'be', 'con', 'de', 'del',
    'did', 'do', 'el', 'en', 'es', 'for', 'from', 'how', 'i', 'la', 'las', 'lo',
    'los', 'me', 'mi', 'mis', 'my', 'of', 'on', 'or', 'para', 'por', 'que', 'the',
    'this', 'to', 'un', 'una', 'what', 'with', 'y'
}


def query_terms(query: str) -> list[str]:
    query = normalize_text(query)
    cleaned = []
    current = []
    for char in query:
        if char.isalnum() or char in {'-', '_'}:
            current.append(char)
        else:
            if current:
                cleaned.append(''.join(current))
                current = []
    if current:
        cleaned.append(''.join(current))
    return [term for term in cleaned if len(term) >= 3 and term not in STOP_WORDS]


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", "ignore").decode("ascii")
    return text.lower()


def load_candidate_records(conn: sqlite3.Connection,
                           category: str = None,
                           source: str = None) -> list[dict]:
    sql = """
        SELECT cr.id, cr.title, cr.category, cr.importance_score,
               cr.created_at, cr.metadata, sf.source_type
        FROM content_records cr
        JOIN source_files sf ON sf.id = cr.source_file_id
        WHERE 1=1
    """
    params = []
    if category:
        sql += " AND cr.category = ?"
        params.append(category)
    if source:
        sql += " AND sf.source_type = ?"
        params.append(source)

    rows = conn.execute(sql, params).fetchall()
    records = []
    for row in rows:
        metadata = json.loads(row[5]) if row[5] else {}
        title = row[1] or ""
        content_text = metadata.get("content_text") or metadata.get("content_preview") or ""
        records.append({
            'id': row[0],
            'title': row[1],
            'category': row[2],
            'importance': row[3],
            'created_at': row[4],
            'metadata': metadata,
            'source': row[6],
            'content_preview': (metadata.get("content_preview") or "")[:500],
            'normalized_text': normalize_text(f"{title}\n{content_text}"),
        })
    return records


def min_term_matches(terms: list[str]) -> int:
    if len(terms) >= 4:
        return 3
    if len(terms) >= 2:
        return 2
    return 1


def score_record_match(normalized_text: str, terms: list[str], base_score: float) -> tuple[float, list[str]]:
    matched = []
    score = base_score or 0.0
    for term in terms:
        if term in normalized_text:
            matched.append(term)
            score += min(normalized_text.count(term) * 0.7, 2.5)
    if not matched:
        return score, matched
    score += len(matched) * 1.8
    score += proximity_bonus(normalized_text, matched)
    return score, matched


def search_keywords(conn: sqlite3.Connection, query: str,
                    category: str = None, source: str = None,
                    limit: int = 20) -> list[dict]:
    """Search for content matching keywords."""
    
    search_terms = query_terms(query)
    records = load_candidate_records(conn, category=category, source=source)
    required_matches = min_term_matches(search_terms) if search_terms else 0
    scored = []

    for record in records:
        score, matched = score_record_match(
            record['normalized_text'],
            search_terms,
            record['importance'],
        )
        if search_terms and len(set(matched)) < required_matches:
            continue
        scored.append({
            'id': record['id'],
            'title': record['title'],
            'category': record['category'],
            'importance': record['importance'],
            'created_at': record['created_at'],
            'metadata': record['metadata'],
            'content_preview': record['content_preview'],
            'score': score,
            'matched_terms': sorted(set(matched)),
        })

    scored.sort(
        key=lambda item: (
            float(item.get('score') or item.get('importance') or 0),
            str(item.get('created_at') or ''),
        ),
        reverse=True,
    )
    deduped = []
    seen = set()
    for item in scored:
        signature = (
            normalize_text(item.get('title') or '')[:120],
            normalize_text(item.get('content_preview') or '')[:120],
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def retrieve_passages(conn: sqlite3.Connection, query: str,
                      category: str = None, source: str = None,
                      limit: int = 8) -> list[dict]:
    """Retrieve relevant grounded passages for local answer generation."""

    terms = query_terms(query)
    records = load_candidate_records(conn, category=category, source=source)
    required_matches = min_term_matches(terms) if terms else 0
    candidate_records = []

    for record in records:
        score, matched = score_record_match(
            record['normalized_text'],
            terms,
            record['importance'],
        )
        if terms and len(set(matched)) < required_matches:
            continue
        candidate_records.append((score, matched, record))

    candidate_records.sort(
        key=lambda item: (item[0], item[2].get('created_at') or ''),
        reverse=True,
    )
    scored = []

    for _, record_matches, record in candidate_records[:max(limit * 12, 24)]:
        metadata = record['metadata']
        content_text = metadata.get('content_text') or metadata.get('content_preview') or ''
        normalized_content = normalize_text(content_text)
        candidate_chunks = conn.execute(
            """SELECT chunk_text FROM content_chunks
               WHERE content_record_id = ?
               ORDER BY chunk_index
               LIMIT 12""",
            (record['id'],)
        ).fetchall()

        for chunk_row in candidate_chunks:
            text = chunk_row[0] or ""
            normalized = normalize_text(text)
            score = record['importance'] or 0
            matched = []
            for term in terms:
                if term in normalized:
                    score += min(normalized.count(term) * 0.8, 3.0)
                    matched.append(term)
                elif term in normalized_content:
                    score += 0.3
                    matched.append(term)
            if not matched and terms:
                continue
            score += proximity_bonus(normalized, matched)
            score -= code_penalty(text)
            score += len(set(matched)) * 1.8
            scored.append({
                'id': record['id'],
                'title': record['title'],
                'category': record['category'],
                'importance': record['importance'],
                'created_at': record['created_at'],
                'metadata': metadata,
                'source': record['source'],
                'chunk_text': text,
                'score': score,
                'matched_terms': sorted(set(matched)),
            })

    scored.sort(key=lambda item: item['score'], reverse=True)
    deduped = []
    seen = set()
    for item in scored:
        signature = (
            item['id'],
            normalize_text(item['chunk_text'])[:220],
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def search_by_category(conn: sqlite3.Connection, category: str,
                       days: int = 30, limit: int = 20,
                       source: str = None) -> list[dict]:
    """Get recent content by category."""
    
    sql = """
        SELECT cr.id, cr.title, cr.category, cr.importance_score,
               cr.created_at, cr.metadata
        FROM content_records cr
        JOIN source_files sf ON sf.id = cr.source_file_id
        WHERE cr.category = ?
        AND cr.created_at >= date('now', ?)
    """

    params = [category, f'-{days} days']
    if source:
        sql += " AND sf.source_type = ?"
        params.append(source)
    sql += " ORDER BY cr.importance_score DESC, cr.created_at DESC LIMIT ?"
    params.append(limit)

    results = conn.execute(sql, params).fetchall()
    
    return [
        {
            'id': r[0],
            'title': r[1],
            'category': r[2],
            'importance': r[3],
            'created_at': r[4],
            'metadata': json.loads(r[5]) if r[5] else {}
        }
        for r in results
    ]


def search_recent(conn: sqlite3.Connection, days: int = 7,
                  limit: int = 20, source: str = None) -> list[dict]:
    """Get recent content across all categories."""
    
    sql = """
        SELECT cr.id, cr.title, cr.category, cr.importance_score,
               cr.created_at, cr.metadata,
               (SELECT chunk_text FROM content_chunks 
                WHERE content_record_id = cr.id LIMIT 1) as preview
        FROM content_records cr
        JOIN source_files sf ON sf.id = cr.source_file_id
        WHERE cr.created_at >= date('now', ?)
    """

    params = [f'-{days} days']
    if source:
        sql += " AND sf.source_type = ?"
        params.append(source)
    sql += " ORDER BY cr.created_at DESC, cr.importance_score DESC LIMIT ?"
    params.append(limit)

    results = conn.execute(sql, params).fetchall()
    
    return [
        {
            'id': r[0],
            'title': r[1],
            'category': r[2],
            'importance': r[3],
            'created_at': r[4],
            'metadata': json.loads(r[5]) if r[5] else {},
            'preview': r[6][:200] if r[6] else ''
        }
        for r in results
    ]


def get_record_detail(conn: sqlite3.Connection, record_id: int) -> dict | None:
    """Get full details of a specific record."""
    
    record = conn.execute(
        """SELECT cr.id, cr.external_id, cr.title, cr.category, 
                  cr.importance_score, cr.created_at, cr.tags, cr.metadata
           FROM content_records cr
           WHERE cr.id = ?""",
        (record_id,)
    ).fetchone()
    
    if not record:
        return None
    
    # Get chunks
    chunks = conn.execute(
        """SELECT chunk_text, chunk_index FROM content_chunks 
           WHERE content_record_id = ? ORDER BY chunk_index""",
        (record_id,)
    ).fetchall()
    
    return {
        'id': record[0],
        'external_id': record[1],
        'title': record[2],
        'category': record[3],
        'importance': record[4],
        'created_at': record[5],
        'tags': json.loads(record[6]) if record[6] else [],
        'metadata': json.loads(record[7]) if record[7] else {},
        'chunks': [c[0] for c in chunks]
    }


def run_query(query: str = None, category: str = None,
              recent_days: int = None, record_id: int = None,
              source: str = None, limit: int = 20,
              db_path: Path = DB_PATH) -> list[dict] | dict:
    """Run a knowledge query."""
    
    conn = sqlite3.connect(db_path)
    
    try:
        if record_id:
            return get_record_detail(conn, record_id)
        
        if recent_days:
            return search_recent(conn, recent_days, limit=limit, source=source)
        
        if category:
            return search_by_category(conn, category, limit=limit, source=source)
        
        if query:
            return search_keywords(conn, query, source=source, limit=limit)
        
        # Default: recent items
        return search_recent(conn, 7, limit=limit, source=source)
        
    finally:
        conn.close()


def answer_with_local_model(question: str, passages: list[dict],
                            base_url: str, model: str,
                            api_key: str | None = None) -> str:
    """Ask a local OpenAI-compatible endpoint using retrieved passages."""

    context_blocks = []
    for index, passage in enumerate(passages, 1):
        chunk_text = (passage['chunk_text'] or '')[:420]
        context_blocks.append(
            "\n".join([
                f"[Passage {index}]",
                f"Source: {passage['source']}",
                f"Title: {passage.get('title') or 'Untitled'}",
                f"Created: {passage.get('created_at') or 'unknown'}",
                chunk_text,
            ])
        )

    payload = {
        'model': model,
        'temperature': 0.1,
        'max_tokens': 180,
        'messages': [
            {
                'role': 'system',
                'content': (
                    "You are a private life/archive assistant. Answer only from the provided passages. "
                    "If the archive context is insufficient, say so clearly. Keep the answer concise and cite passage numbers."
                ),
            },
            {
                'role': 'user',
                'content': f"Question:\n{question}\n\nRetrieved archive passages:\n\n" + "\n\n".join(context_blocks),
            },
        ],
    }

    req = request.Request(
        base_url.rstrip('/') + '/chat/completions',
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            **({'Authorization': f'Bearer {api_key}'} if api_key else {}),
        },
        method='POST',
    )

    try:
        with request.urlopen(req, timeout=180) as response:
            body = json.loads(response.read().decode('utf-8'))
    except error.HTTPError as exc:
        details = exc.read().decode('utf-8', errors='replace')
        raise RuntimeError(f"Local model request failed: {exc.code} {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Local model request failed: {exc}") from exc

    choices = body.get('choices') or []
    if not choices:
        raise RuntimeError("Local model response contained no choices")
    message = choices[0].get('message', {})
    content = message.get('content', '')
    if isinstance(content, list):
        content = "\n".join(part.get('text', '') if isinstance(part, dict) else str(part) for part in content)
    return str(content).strip()


def proximity_bonus(text: str, matched_terms: list[str]) -> float:
    positions = [text.find(term) for term in matched_terms if term in text]
    if len(positions) < 2:
        return 0.0
    span = max(positions) - min(positions)
    return max(0.0, 4.0 - (span / 220.0))


def code_penalty(text: str) -> float:
    lowered = text.lower()
    hits = 0
    for needle in ['```', '/home/', '.ts', '.kt', 'const ', 'function ', '=>', './']:
        hits += lowered.count(needle)
    return min(hits * 0.4, 4.0)


def print_results(results: list[dict] | dict, format: str = 'text'):
    """Print query results."""
    
    if isinstance(results, dict) and 'chunks' in results:
        # Single record detail
        print(f"\n=== Record {results['id']} ===")
        print(f"Title: {results['title']}")
        print(f"Category: {results['category']}")
        print(f"Importance: {results['importance']:.2f}")
        print(f"Created: {results['created_at']}")
        print(f"\nContent ({len(results['chunks'])} chunks):")
        for i, chunk in enumerate(results['chunks'][:3]):
            print(f"\n[Chunk {i+1}]\n{chunk[:500]}...")
        return
    
    if not results:
        print("No results found")
        return
    
    print(f"\nFound {len(results)} results:\n")
    
    for i, r in enumerate(results, 1):
        if format == 'json':
            print(json.dumps(r, indent=2))
        else:
            print(f"{i}. [{r['category']}] {(r['title'] or 'Untitled')[:60]}...")
            created_at = r.get('created_at') or 'unknown'
            display_score = r.get('score', r.get('importance', 0))
            print(f"   Score: {display_score:.2f} | Created: {str(created_at)[:10]}")
            preview = r.get('preview') or r.get('content_preview')
            if preview:
                print(f"   Preview: {preview[:100]}...")
            print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Query knowledge base")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--category", "-c", help="Filter by category")
    parser.add_argument("--source", "-s", help="Filter by source type")
    parser.add_argument("--recent", "-r", type=int, help="Recent N days")
    parser.add_argument("--id", "-i", type=int, help="Get specific record ID")
    parser.add_argument("--answer", action="store_true", help="Ask a local model using retrieved passages")
    parser.add_argument("--top", type=int, default=8, help="Top retrieved passages for --answer")
    parser.add_argument("--base-url", default="http://127.0.0.1:8082/v1", help="Local OpenAI-compatible base URL")
    parser.add_argument("--model", default="local", help="Local model name")
    parser.add_argument("--api-key", help="Optional API key for local endpoint")
    parser.add_argument("--json", "-j", action="store_true", help="JSON output")
    parser.add_argument("--limit", "-l", type=int, default=20, help="Result limit")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Database path")
    
    args = parser.parse_args()
    
    if args.answer:
        if not args.query:
            print("--answer requires a query string", file=sys.stderr)
            sys.exit(1)
        conn = sqlite3.connect(args.db)
        try:
            passages = retrieve_passages(
                conn,
                args.query,
                category=args.category,
                source=args.source,
                limit=args.top,
            )
        finally:
            conn.close()

        if not passages:
            print("No relevant passages found")
            sys.exit(0)

        print(f"\nRetrieved {len(passages)} passages:\n")
        for idx, passage in enumerate(passages, 1):
            preview = passage['chunk_text'][:180].replace('\n', ' ')
            print(f"{idx}. [{passage['source']}] {(passage.get('title') or 'Untitled')[:60]}")
            print(f"   Score: {passage['score']:.2f} | Created: {(passage.get('created_at') or 'unknown')[:19]}")
            print(f"   Preview: {preview}...")
            print()

        answer = answer_with_local_model(
            args.query,
            passages,
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
        )
        print("Answer:\n")
        print(answer)
    else:
        results = run_query(
            query=args.query,
            category=args.category,
            recent_days=args.recent,
            record_id=args.id,
            source=args.source,
            limit=args.limit,
            db_path=args.db
        )
        
        print_results(results, format='json' if args.json else 'text')
