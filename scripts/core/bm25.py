#!/usr/bin/env python3
"""
BM25 retriever for RAG - proper term frequency, IDF, and length normalization.
Caches the index in memory for fast repeated queries.
"""
import sys
import sqlite3
import math
import re
from pathlib import Path
from threading import Lock

sys.stdout.reconfigure(encoding='utf-8')

DB_PATH = Path("data/state.db")

STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'you', 'your', 'he', 'him', 'his',
    'she', 'her', 'it', 'its', 'they', 'them', 'their', 'what', 'which', 'who', 'whom',
    'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
    'by', 'for', 'with', 'about', 'against', 'between', 'through', 'during', 'before',
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
    'el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'en', 'y', 'o', 'que', 'por',
    'para', 'con', 'sin', 'se', 'lo', 'mi', 'tu', 'su', 'es', 'son', 'tiene', 'han',
    'work', 'worked', 'working', 'done', 'know', 'tell', 'get', 'got', 'make', 'made',
    'see', 'saw', 'say', 'said', 'go', 'went', 'come', 'came', 'think', 'thought',
    'take', 'took', 'find', 'found', 'give', 'gave', 'use', 'used', 'try', 'tried',
    'need', 'called', 'call', 'set', 'put', 'want', 'run', 'ran', 'move', 'like',
    'also', 'back', 'even', 'well', 'way', 'many', 'much', 'into', 'over', 'after',
    'any', 'thing', 'things', 'stuff', 'project', 'projects', 'something',
}

def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r'[a-z0-9]{3,}', text)
    return [t for t in tokens if t not in STOP_WORDS]


class BM25Retriever:
    """BM25 retriever with in-memory inverted index."""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.k1 = 1.5
        self.b = 0.75
        self.avg_doc_len = 0
        self.N = 0
        self.idf = {}
        self.documents = []
        self._build_index(db_path)
    
    def _build_index(self, db_path: Path):
        if not db_path.exists():
            return
        
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                """SELECT cc.id, cc.chunk_text, cr.title, sf.source_type, cr.created_at
                   FROM content_chunks cc
                   JOIN content_records cr ON cc.content_record_id = cr.id
                   JOIN source_files sf ON cr.source_file_id = sf.id"""
            ).fetchall()
        finally:
            conn.close()
        
        for row in rows:
            chunk_id, text, title, source, created_at = row
            if not text or len(text) < 20:
                continue
            tokens = tokenize(text)
            if not tokens:
                continue
            self.documents.append({
                'id': chunk_id,
                'text': text[:500],
                'title': title or 'Untitled',
                'source': source,
                'created_at': created_at or 'unknown',
                'tokens': tokens,
            })
        
        self.N = len(self.documents)
        if self.N == 0:
            return
        
        # Compute document frequency
        term_doc_freq = {}
        total_tokens = 0
        for doc in self.documents:
            total_tokens += len(doc['tokens'])
            seen = set()
            for t in doc['tokens']:
                if t not in seen:
                    term_doc_freq[t] = term_doc_freq.get(t, 0) + 1
                    seen.add(t)
        
        self.avg_doc_len = total_tokens / self.N
        
        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        for term, df in term_doc_freq.items():
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
    
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.N == 0:
            return []
        
        # Expand query with related terms
        expanded = expand_query(query)
        query_tokens = tokenize(expanded)
        if not query_tokens:
            return []
        
        # Also parse original query for tool/task matching
        orig_tokens = tokenize(query)
        
        scores = []
        for doc in self.documents:
            score = 0.0
            doc_len = len(doc['tokens'])
            doc_term_freq = {}
            for t in doc['tokens']:
                doc_term_freq[t] = doc_term_freq.get(t, 0) + 1
            
            # BM25 scoring on expanded query
            for qt in query_tokens:
                if qt not in doc_term_freq:
                    continue
                tf = doc_term_freq[qt]
                idf = self.idf.get(qt, 0.0)
                if idf <= 0:
                    continue
                num = tf * (self.k1 + 1)
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                score += idf * num / denom
            
            # Bonus for matching original query tokens more strongly
            for qt in orig_tokens:
                if qt in doc_term_freq:
                    score += 2.0  # Bonus for exact match
            
            if score > 0:
                scores.append((score, doc))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, doc in scores[:top_k]:
            results.append({
                'id': doc['id'],
                'text': doc['text'],
                'title': doc['title'],
                'source': doc['source'],
                'created_at': doc['created_at'],
                'score': round(score, 3),
            })
        
        return results


# Global cached instance
_retriever = None
_retriever_lock = Lock()

def get_retriever(db_path: Path = DB_PATH) -> BM25Retriever:
    """Get or create the cached BM25 retriever."""
    global _retriever
    with _retriever_lock:
        if _retriever is None:
            _retriever = BM25Retriever(db_path)
        return _retriever

def retrieve_chunks(query: str, limit: int = 5, db_path: Path = DB_PATH) -> list[dict]:
    """Public API - retrieve chunks using cached BM25 index."""
    retriever = get_retriever(db_path)
    return retriever.search(query, limit)


# Query expansion patterns for common questions
QUERY_EXPANSIONS = {
    "age": ["birth", "born", "year", "old"],
    "project": ["repo", "github", "code", "app", "bot", "service"],
    "work": ["task", "build", "implement", "fix", "feature", "code"],
    "fan": ["pwm", "thermal", "cooling", "temperature", "heat"],
    "model": ["llm", "ai", "qwen", "gpt", "inference", "gguf"],
}

def expand_query(query: str) -> str:
    """Add related terms for better BM25 recall."""
    q_lower = query.lower()
    expanded = query
    for keyword, related_terms in QUERY_EXPANSIONS.items():
        if keyword in q_lower:
            expanded += " " + " ".join(related_terms[:3])
            break
    return expanded
