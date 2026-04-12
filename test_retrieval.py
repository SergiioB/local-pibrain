#!/usr/bin/env python3
"""Test retrieval quality across diverse queries."""
import urllib.request, json, time, sys

QUERIES = [
    ('EN-1', 'what GPU hardware do I have'),
    ('EN-2', 'how to deploy with Docker'),
    ('EN-3', 'my Android projects'),
    ('ES-1', 'que hardware tengo'),
    ('ES-2', 'configuracion de la base de datos'),
    ('ES-3', 'problemas con el disco duro'),
    ('EN-4', 'fan control rock 5b temperature'),
    ('EN-5', 'my AI toolchain and development setup'),
    ('ES-4', 'como configurar el servidor local'),
    ('EN-6', 'networking and server configuration'),
]

results = []
for label, q in QUERIES:
    req = urllib.request.Request(
        'http://localhost:8096/api/chat',
        data=json.dumps({'question': q, 'generate_answer': False, 'top_k': 5}).encode(),
        headers={'Content-Type': 'application/json'})
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read())
    passages = data.get('passages', [])
    results.append((label, q, passages))
    time.sleep(0.2)

for label, q, passages in results:
    print(f'=== {label}: "{q}" ===')
    print(f'  Found: {len(passages)} passages')
    for i, p in enumerate(passages[:3]):
        text_preview = p['text'][:80].replace('\n', ' ')
        print(f'  [{i+1}] [{p["source"]}] {p["title"][:60]} score={p["score"]} ({p["created_at"]})')
        print(f'      {text_preview}...')
    if not passages:
        print('  *** NO RESULTS ***')
    print()
