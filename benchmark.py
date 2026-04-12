#!/usr/bin/env python3
"""Benchmark the LocalBrain RAG pipeline with diverse questions."""
import json
import time
import sys
import urllib.request

sys.stdout.reconfigure(encoding='utf-8')

BASE = "http://127.0.0.1:8096"

TESTS = [
    # Category, question
    ("personal", "What is my name?"),
    ("personal", "Where do I live?"),
    ("factual", "What hardware do I use?"),
    ("factual", "What devices am I working with?"),
    ("project", "Tell me about my Discord bot project"),
    ("project", "What AI tools have I used?"),
    ("project", "What coding sessions do I have from recent months?"),
    ("technical", "How does fan control work on my device?"),
    ("technical", "What models am I running locally?"),
    ("technical", "Tell me about my OpenCode setup"),
    ("technical", "How do I use llama-server?"),
    ("search", "Python and OpenAI API"),
    ("search", "AI assistant sessions"),
    ("search", "Discord bot"),
    ("search", "Pi sessions"),
    ("ambiguous", "What can you do?"),
    ("ambiguous", "How was I today?"),
    ("improve", "How can I make my RAG system better?"),
    ("improve", "How to improve my fan control script?"),
    ("improve", "Give me ideas to improve my workflow"),
]

def ask(question, generate=True, top_k=5):
    data = json.dumps({
        "question": question,
        "generate_answer": generate,
        "top_k": top_k,
    }).encode()
    req = urllib.request.Request(
        BASE + "/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=120)
    result = json.loads(resp.read().decode())
    elapsed = time.perf_counter() - t0
    result["_elapsed_real"] = round(elapsed, 2)
    return result

def score_result(result):
    """Score the response quality."""
    n_passages = len(result.get("passages", []))
    has_llm = bool(result.get("answer"))
    no_error = "error" not in result.get("answer", "").lower()
    not_hallucinated = "i don't have" not in result.get("answer", "").lower()
    score = 0
    if n_passages > 0:
        score += 1
    if n_passages >= 3:
        score += 1
    if has_llm and no_error:
        score += 1
    if not_hallucinated or n_passages == 0:
        score += 1
    return min(score, 4)

print("=" * 70)
print("LocalBrain RAG Benchmark")
print("=" * 70)

total_score = 0
total_tests = len(TESTS)

for i, (category, question) in enumerate(TESTS, 1):
    try:
        result = ask(question, generate=True, top_k=5)
        score = score_result(result)
        total_score += score

        passages = len(result.get("passages", []))
        elapsed = result.get("_elapsed_real", result.get("elapsed_ms", 0) / 1000)
        answer_preview = result.get("answer", "")[:120]

        print(f"\n[{i}/{total_tests}] [{category}] Score: {score}/4")
        print(f"  Q: {question}")
        print(f"  Passages: {passages} | Time: {elapsed:.1f}s")
        print(f"  A: {answer_preview}")
        print(f"  Top passage scores: {[p.get('score', 0) for p in result.get('passages', [])[:3]]}")
    except Exception as e:
        print(f"\n[{i}/{total_tests}] [{category}] FAIL: {e}")

print(f"\n{'=' * 70}")
print(f"Overall: {total_score}/{total_tests * 4} ({total_score/(total_tests*4)*100:.0f}%)")
print(f"{'=' * 70}")
