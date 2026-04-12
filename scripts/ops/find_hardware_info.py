import sqlite3, sys
sys.stdout.reconfigure(encoding='utf-8')
c = sqlite3.connect(r'C:\Users\Sergiio\Syncthing\localbrain\data\state.db')

# Search for RTX 5070 specifically
print("=== RTX 5070 mentions ===")
rows = c.execute("""SELECT rowid, title, source_type, SUBSTR(chunk_text, 1, 400) 
                    FROM chunks_fts WHERE chunks_fts MATCH 'rtx 5070' 
                    ORDER BY rank LIMIT 5""").fetchall()
for r in rows:
    print(f"  [{r[2]}] {r[1][:50]}")
    print(f"    {r[3][:300].replace(chr(10), ' ')}")
    print()

# Search for Ryzen
print("=== Ryzen 5700x ===")
rows = c.execute("""SELECT rowid, title, source_type, SUBSTR(chunk_text, 1, 400)
                    FROM chunks_fts WHERE chunks_fts MATCH 'ryzen 5700x'
                    ORDER BY rank LIMIT 5""").fetchall()
for r in rows:
    print(f"  [{r[2]}] {r[1][:50]}")
    print(f"    {r[3][:300].replace(chr(10), ' ')}")
    print()

# Search for 'tengo' AND 'rtx'
print("=== tengo AND rtx ===")
rows = c.execute("""SELECT rowid, title, source_type, SUBSTR(chunk_text, 1, 400)
                    FROM chunks_fts WHERE chunks_fts MATCH 'tengo AND rtx'
                    ORDER BY rank LIMIT 5""").fetchall()
for r in rows:
    print(f"  [{r[2]}] {r[1][:50]}")
    print(f"    {r[3][:300].replace(chr(10), ' ')}")
    print()

# Search for 'workstation' 
print("=== workstation ===")
rows = c.execute("""SELECT rowid, title, source_type, SUBSTR(chunk_text, 1, 400)
                    FROM chunks_fts WHERE chunks_fts MATCH 'workstation'
                    ORDER BY rank LIMIT 5""").fetchall()
for r in rows:
    print(f"  [{r[2]}] {r[1][:50]}")
    print(f"    {r[3][:300].replace(chr(10), ' ')}")
    print()

# Check content_records titles that might have hardware info
print("=== Record titles with hardware context ===")
rows = c.execute("""SELECT cr.id, cr.title, cr.source_file_id, sf.source_type
                    FROM content_records cr
                    JOIN source_files sf ON sf.id = cr.source_file_id
                    WHERE cr.title LIKE '%hardware%' OR cr.title LIKE '%spec%' 
                    OR cr.title LIKE '%gpu%' OR cr.title LIKE '%rtx%'
                    OR cr.title LIKE '%setup%'
                    LIMIT 20""").fetchall()
for r in rows:
    print(f"  [{r[3]}] id={r[0]} {r[1][:70]}")

c.close()
