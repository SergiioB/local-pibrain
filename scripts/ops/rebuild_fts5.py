#!/usr/bin/env python3
"""Rebuild FTS5 table with all needed columns including content_record_id."""
import sqlite3
import time
import sys

sys.stdout.reconfigure(encoding='utf-8')

DB = r'C:\Users\Sergiio\Syncthing\localbrain\data\state.db'

print('Connecting to database...', flush=True)
c = sqlite3.connect(DB)

# Drop old FTS5
print('Dropping old FTS5 table...', flush=True)
c.execute('DROP TABLE IF EXISTS chunks_fts')
c.commit()

# Create standalone FTS5 with all columns needed
print('Creating standalone FTS5 table with content_record_id...', flush=True)
c.execute('''
    CREATE VIRTUAL TABLE chunks_fts USING fts5(
        chunk_text,
        title,
        source_type,
        created_at,
        content_record_id,
        tokenize='unicode61'
    )
''')
c.commit()

# Populate
print('Populating FTS5...', flush=True)
t0 = time.time()
cursor = c.execute('''
    SELECT cc.chunk_text, cr.title, sf.source_type, cr.created_at, cc.content_record_id
    FROM content_chunks cc
    JOIN content_records cr ON cc.content_record_id = cr.id
    JOIN source_files sf ON cr.source_file_id = sf.id
''')

batch = []
total = 0
for row in cursor:
    text = (row[0] or '').replace('\n', ' ').replace('\r', ' ')[:2000]
    batch.append((
        text,
        row[1] or '',
        row[2] or '',
        str(row[3]) if row[3] else '',
        row[4] or 0
    ))
    total += 1
    if len(batch) >= 5000:
        c.executemany('INSERT INTO chunks_fts VALUES (?, ?, ?, ?, ?)', batch)
        c.commit()
        print(f'  {total} rows...', flush=True)
        batch = []

if batch:
    c.executemany('INSERT INTO chunks_fts VALUES (?, ?, ?, ?, ?)', batch)
    c.commit()

elapsed = time.time() - t0
cnt = c.execute('SELECT count(*) FROM chunks_fts').fetchone()[0]
print(f'Done: {cnt} entries in {elapsed:.1f}s', flush=True)

# Test
print('Testing FTS5 query with content_record_id...', flush=True)
rows = c.execute(
    "SELECT rowid, title, source_type, created_at, content_record_id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT 3",
    ("gpu hardware",)
).fetchall()
for r in rows:
    print(f'  [{r[2]}] {r[1][:60]} rid={r[4]}', flush=True)

c.close()
print('FTS5 rebuilt successfully with content_record_id.', flush=True)
