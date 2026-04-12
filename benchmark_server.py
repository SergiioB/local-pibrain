#!/usr/bin/env python3
"""Benchmark the optimized web server with English and Spanish queries."""
import time
import urllib.request
import json

BASE = "http://localhost:8096"

def query(question, generate=False, top_k=5):
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{BASE}/api/chat",
        data=json.dumps({"question": question, "generate_answer": generate, "top_k": top_k}).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=60)
    data = json.loads(resp.read())
    wall_ms = (time.perf_counter() - t0) * 1000
    return data, wall_ms

# Status check
t0 = time.perf_counter()
req = urllib.request.Request(f"{BASE}/api/status")
resp = urllib.request.urlopen(req, timeout=10)
status = json.loads(resp.read())
print(f"Status: {status['records']} records, {status['chunks']} chunks")
print(f"Status call: {(time.perf_counter()-t0)*1000:.0f}ms\n")

queries = [
    ("EN-1", "what is my project about"),
    ("EN-2", "what is my project about"),       # cached
    ("EN-3", "how does the AI model work"),
    ("EN-4", "hardware configuration gpu memory"),
    ("ES-1", "cuales son mis proyectos principales"),
    ("ES-2", "configuracion del hardware gpu memoria"),
    ("ES-3", "como funciona la inteligencia artificial local"),
    ("EN-5", "docker container deployment setup"),
    ("ES-4", "base de datos sqlite configuracion"),
    ("EN-1-repeat", "what is my project about"),  # cached again
]

print(f"{'Query':<15} {'Passages':>8} {'Server ms':>10} {'Wall ms':>10} {'Top title'}")
print("-" * 90)

times = []
for label, q in queries:
    data, wall_ms = query(q, generate=False, top_k=5)
    server_ms = data["elapsed_ms"]
    n_passages = len(data["passages"])
    top_title = data["passages"][0]["title"][:50] if data["passages"] else "(none)"
    times.append(wall_ms)
    print(f"{label:<15} {n_passages:>8} {server_ms:>10.1f} {wall_ms:>10.1f} {top_title}")

print("-" * 90)
print(f"Average wall time: {sum(times)/len(times):.0f}ms")
print(f"Min wall time:   {min(times):.0f}ms")
print(f"Max wall time:   {max(times):.0f}ms")
print(f"Cached queries (EN-2, EN-1-repeat): check if faster above")
