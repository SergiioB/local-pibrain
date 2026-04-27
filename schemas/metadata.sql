-- SQLite schema for personal AI node metadata and state
-- Schema version: 1.0.0

-- Source file manifest
CREATE TABLE IF NOT EXISTS source_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    source_type TEXT NOT NULL,  -- droid, opencode, codex, gemini_cli, etc.
    file_hash TEXT NOT NULL,
    file_size_bytes INTEGER,
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_ingested_at TIMESTAMP,
    ingestion_status TEXT DEFAULT 'pending',  -- pending, ingested, failed, skipped
    record_count INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    error_message TEXT,
    metadata JSON  -- source-specific metadata
);

CREATE INDEX idx_source_files_status ON source_files(ingestion_status);
CREATE INDEX idx_source_files_type ON source_files(source_type);
CREATE INDEX idx_source_files_discovered ON source_files(discovered_at);

-- Extracted content records
CREATE TABLE IF NOT EXISTS content_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file_id INTEGER NOT NULL,
    record_type TEXT NOT NULL,  -- conversation, message, document, note
    external_id TEXT,  -- original ID from source
    title TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    category TEXT,  -- code, chat, tool, reasoning
    tags JSON,
    importance_score REAL DEFAULT 0.5,
    metadata JSON,
    FOREIGN KEY (source_file_id) REFERENCES source_files(id)
);

CREATE INDEX idx_content_records_source ON content_records(source_file_id);
CREATE INDEX idx_content_records_type ON content_records(record_type);
CREATE INDEX idx_content_records_category ON content_records(category);
CREATE INDEX idx_content_records_created ON content_records(created_at);

-- Chunked content for retrieval
CREATE TABLE IF NOT EXISTS content_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_record_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_hash TEXT NOT NULL,
    start_char INTEGER,
    end_char INTEGER,
    embedding_status TEXT DEFAULT 'pending',  -- pending, embedded, failed
    embedding_model TEXT,
    metadata JSON,
    FOREIGN KEY (content_record_id) REFERENCES content_records(id),
    UNIQUE(content_record_id, chunk_index)
);

CREATE INDEX idx_chunks_record ON content_chunks(content_record_id);
CREATE INDEX idx_chunks_status ON content_chunks(embedding_status);

-- Embedding vectors (if sqlite-vec available)
-- Note: sqlite-vec extension must be loaded before creating
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding FLOAT[384]  -- all-MiniLM-L6-v2 dimensions
);

-- Morning briefing history
CREATE TABLE IF NOT EXISTS briefing_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    briefing_date DATE NOT NULL,
    content_markdown TEXT NOT NULL,
    item_count INTEGER,
    sources_used JSON,
    generation_time_ms INTEGER,
    metadata JSON
);

CREATE INDEX idx_briefing_date ON briefing_history(briefing_date);

-- Briefing items (for tracking what appeared)
CREATE TABLE IF NOT EXISTS briefing_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    briefing_id INTEGER NOT NULL,
    content_record_id INTEGER,
    item_type TEXT NOT NULL,  -- recent_note, resurfaced, follow_up, language_review, approval
    score REAL NOT NULL,
    score_breakdown JSON,
    title TEXT,
    summary TEXT,
    action_required BOOLEAN DEFAULT FALSE,
    shown_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    action_taken_at TIMESTAMP,
    action_type TEXT,  -- viewed, dismissed, acted_on
    FOREIGN KEY (briefing_id) REFERENCES briefing_history(id),
    FOREIGN KEY (content_record_id) REFERENCES content_records(id)
);

-- arXiv paper tracking
CREATE TABLE IF NOT EXISTS arxiv_papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    arxiv_id TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    title_normalized TEXT,  -- for dedup
    authors JSON,
    categories JSON,
    abstract TEXT,
    published_date DATE,
    updated_date DATE,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    relevance_score REAL,
    importance_score REAL,
    novelty_score REAL,
    combined_score REAL,
    approval_status TEXT DEFAULT 'pending',  -- pending, approved, rejected, ingested
    approved_at TIMESTAMP,
    rejected_at TIMESTAMP,
    rejection_reason TEXT,
    ingested_at TIMESTAMP,
    source_url TEXT,
    metadata JSON
);

CREATE INDEX idx_arxiv_status ON arxiv_papers(approval_status);
CREATE INDEX idx_arxiv_score ON arxiv_papers(combined_score DESC);
CREATE INDEX idx_arxiv_date ON arxiv_papers(published_date);

-- Portfolio draft tracking
CREATE TABLE IF NOT EXISTS portfolio_drafts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    draft_slug TEXT NOT NULL UNIQUE,  -- filename slug for portfolio post
    suggested_title TEXT NOT NULL,
    suggested_category TEXT,
    suggested_tags JSON,
    opportunity_score REAL,
    draft_content TEXT,
    draft_markdown_path TEXT,  -- path to generated .md file
    preview_path TEXT,  -- path in portfolio preview
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    preview_generated_at TIMESTAMP,
    approval_status TEXT DEFAULT 'pending',  -- pending, approved, rejected, published
    approved_at TIMESTAMP,
    rejected_at TIMESTAMP,
    rejection_reason TEXT,
    published_at TIMESTAMP,
    published_url TEXT,
    git_commit_sha TEXT,
    feedback_notes JSON,
    source_topics JSON,  -- what topics led to this draft
    source_records JSON  -- which content records influenced it
);

CREATE INDEX idx_drafts_status ON portfolio_drafts(approval_status);
CREATE INDEX idx_drafts_score ON portfolio_drafts(opportunity_score DESC);

-- Approval workflow events
CREATE TABLE IF NOT EXISTS approval_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,  -- arxiv_paper, portfolio_draft
    entity_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,  -- created, previewed, approved, rejected, published
    event_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    actor TEXT DEFAULT 'system',  -- system, user, script
    notes TEXT,
    metadata JSON
);

CREATE INDEX idx_approval_events_entity ON approval_events(entity_type, entity_id);
CREATE INDEX idx_approval_events_type ON approval_events(event_type);

-- Ingestion batches
CREATE TABLE IF NOT EXISTS ingestion_batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_type TEXT NOT NULL,  -- full, incremental
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status TEXT DEFAULT 'running',  -- running, completed, failed, partial
    files_processed INTEGER DEFAULT 0,
    records_created INTEGER DEFAULT 0,
    chunks_created INTEGER DEFAULT 0,
    embeddings_created INTEGER DEFAULT 0,
    errors JSON,
    duration_seconds INTEGER
);

-- Workflow state (key-value for runtime state)
CREATE TABLE IF NOT EXISTS workflow_state (
    key TEXT PRIMARY KEY,
    value JSON,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Default workflow state
INSERT OR IGNORE INTO workflow_state (key, value) VALUES 
    ('last_ingestion', '{"timestamp": null, "batch_id": null}'),
    ('last_briefing', '{"timestamp": null, "briefing_id": null}'),
    ('last_arxiv_fetch', '{"timestamp": null, "papers_fetched": 0}'),
    ('arxiv_cooldown', '{"rejected_ids": [], "updated_at": null}');

-- Retrieval quality metrics
CREATE TABLE IF NOT EXISTS retrieval_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    query_type TEXT,
    strategy TEXT,
    bm25_candidates INTEGER,
    vector_candidates INTEGER,
    final_count INTEGER,
    elapsed_ms REAL,
    top_score REAL,
    avg_score REAL,
    score_spread REAL,
    sources_count INTEGER,
    timestamp TEXT
);

CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON retrieval_metrics(timestamp);

-- Schema metadata
CREATE TABLE IF NOT EXISTS schema_version (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

INSERT OR IGNORE INTO schema_version (version, notes) VALUES 
    ('1.0.0', 'Initial schema for personal AI node MVP');

INSERT OR IGNORE INTO schema_version (version, notes) VALUES 
    ('1.1.0', 'Added retrieval quality metrics, smart chunking, hybrid retrieval');
