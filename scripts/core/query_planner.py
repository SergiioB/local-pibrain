#!/usr/bin/env python3
"""
Query planner for LocalBrain.

Analyzes incoming queries to choose the optimal retrieval strategy.
Based on the video's key insight: different query types need different approaches.

Query types:
  - FACTUAL: "What is X?" → BM25 for exact keyword match
  - ANALYTICAL: "Compare X and Y" → Long context / more passages
  - EXPLORATORY: "Tell me about my projects" → Broad BM25 + semantic
  - PERSONAL: "What hardware do I have?" → Profile lookup first
  - TEMPORAL: "What did I work on last week?" → Recency-weighted BM25
  - COMPARATIVE: "What's missing between X and Y?" → Long context (whole book problem)

This prevents the "silent failure" the video describes: using the wrong
retrieval strategy means the right data exists but never reaches the model.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class QueryType(Enum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    EXPLORATORY = "exploratory"
    PERSONAL = "personal"
    TEMPORAL = "temporal"
    COMPARATIVE = "comparative"
    CODE = "code"
    UNKNOWN = "unknown"


@dataclass
class QueryPlan:
    """Planned retrieval strategy for a query."""
    query_type: QueryType
    strategy: str          # auto, bm25_only, vector_only, long_context, hybrid
    top_k: int             # How many passages to retrieve
    use_reranker: bool     # Whether to cross-encoder rerank
    use_vectors: bool      # Whether to use vector search
    expanded_queries: List[str]  # Query variants
    recency_boost: bool    # Weight recent results higher
    confidence: float      # How confident we are in this plan

    def to_dict(self) -> dict:
        return {
            "query_type": self.query_type.value,
            "strategy": self.strategy,
            "top_k": self.top_k,
            "use_reranker": self.use_reranker,
            "use_vectors": self.use_vectors,
            "expanded_queries": self.expanded_queries,
            "recency_boost": self.recency_boost,
            "confidence": self.confidence,
        }


# ─── Pattern matching for query classification ──────────────────────────

PERSONAL_PATTERNS = [
    r'\b(i|my|me|mine|we|our)\b',
    r'\b(have|i.?ve|got|own|use|running)\b',
    r'\b(tengo|mi |mis |mío|nuestro)\b',
    r'\b(my (hardware|setup|projects|tools|config|devices))\b',
]

TEMPORAL_PATTERNS = [
    r'\b(recent|latest|last|yesterday|today|this week|this month)\b',
    r'\b(recentemente|último|ayer|hoy|esta semana)\b',
    r'\b(past|ago|before|since|new)\b',
    r'\b\d+\s+(days?|weeks?|months?|hours?)\s+ago\b',
]

COMPARATIVE_PATTERNS = [
    r'\b(compare|versus|vs|difference|between)\b',
    r'\b(missing|omitted|gap|lacks?)\b',
    r'\b(what.*(not|isn\'t|wasn\'t))\b',
    r'\b(comparar|diferencia|entre)\b',
]

ANALYTICAL_PATTERNS = [
    r'\b(analyze|explain|how does|why does|summarize)\b',
    r'\b(overview|summary|breakdown|insights?)\b',
    r'\b(analizar|explicar|resumir|por qué)\b',
    r'\b(tell me about|describe|overview)\b',
]

CODE_PATTERNS = [
    r'\b(code|function|class|method|api|endpoint|module)\b',
    r'\b(bug|error|fix|debug|implementation)\b',
    r'\b(import|def |class |const |var |let )\b',
    r'\b(stack.?trace|exception|traceback)\b',
]

FACTUAL_PATTERNS = [
    r'\b(what is|who is|where is|when (did|was|is))\b',
    r'\b(how (to|do|many|much))\b',
    r'\b(define|definition)\b',
    r'\b(cuando|donde|quien|cuanto|cuál)\b',
]


def classify_query(query: str) -> Tuple[QueryType, float]:
    """Classify query type with confidence score.

    Multi-pass classification: first check specific patterns,
    then personal as fallback only if no stronger signal.
    """
    q_lower = query.lower()
    scores = {}

    for qtype, patterns in [
        (QueryType.TEMPORAL, TEMPORAL_PATTERNS),
        (QueryType.COMPARATIVE, COMPARATIVE_PATTERNS),
        (QueryType.ANALYTICAL, ANALYTICAL_PATTERNS),
        (QueryType.CODE, CODE_PATTERNS),
        (QueryType.FACTUAL, FACTUAL_PATTERNS),
        (QueryType.PERSONAL, PERSONAL_PATTERNS),  # Last: personal is weakest signal
    ]:
        score = 0.0
        for pattern in patterns:
            matches = len(re.findall(pattern, q_lower))
            if matches:
                score += matches * 0.3
        scores[qtype] = score

    if not scores or max(scores.values()) < 0.1:
        return QueryType.EXPLORATORY, 0.3

    # If personal ties with a stronger type, prefer the stronger type
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    # Personal is a very weak signal (just pronouns) - only win if clearly dominant
    if best_type == QueryType.PERSONAL:
        # Check if another type has a comparable score
        other_max = max((v for k, v in scores.items() if k != QueryType.PERSONAL), default=0)
        if other_max >= best_score * 0.7 and other_max > 0:
            best_type = max(
                ((k, v) for k, v in scores.items() if k != QueryType.PERSONAL),
                key=lambda x: x[1]
            )[0]
            best_score = scores[best_type]

    confidence = min(best_score, 1.0)
    return best_type, confidence


def plan_query(query: str, default_top_k: int = 10) -> QueryPlan:
    """Analyze a query and create an optimal retrieval plan.

    Maps query types to retrieval strategies based on the video's insights:
      - COMPARATIVE → long_context (solves "whole book problem")
      - FACTUAL → bm25_only (fast, precise keyword match)
      - TEMPORAL → hybrid with recency boost
      - ANALYTICAL → hybrid with reranking
      - PERSONAL → hybrid (may find info in profile or knowledge base)
      - CODE → bm25_only (code has specific keywords)
      - EXPLORATORY → hybrid broad search
    """
    query_type, confidence = classify_query(query)

    # Base plan by query type
    if query_type == QueryType.COMPARATIVE:
        plan = QueryPlan(
            query_type=query_type,
            strategy="long_context",
            top_k=default_top_k * 3,      # More context for comparison
            use_reranker=True,            # Need precise ranking
            use_vectors=True,             # Semantic helps find related docs
            expanded_queries=[query],
            recency_boost=False,
            confidence=confidence,
        )

    elif query_type == QueryType.FACTUAL:
        plan = QueryPlan(
            query_type=query_type,
            strategy="bm25_only",
            top_k=default_top_k,
            use_reranker=False,           # BM25 is sufficient for facts
            use_vectors=False,            # Keywords are precise enough
            expanded_queries=[query],
            recency_boost=False,
            confidence=confidence,
        )

    elif query_type == QueryType.TEMPORAL:
        plan = QueryPlan(
            query_type=query_type,
            strategy="hybrid",
            top_k=default_top_k,
            use_reranker=False,
            use_vectors=True,
            expanded_queries=_temporal_expansions(query),
            recency_boost=True,           # Key: weight recent content
            confidence=confidence,
        )

    elif query_type == QueryType.ANALYTICAL:
        plan = QueryPlan(
            query_type=query_type,
            strategy="hybrid",
            top_k=default_top_k * 2,      # More context for analysis
            use_reranker=True,
            use_vectors=True,
            expanded_queries=[query],
            recency_boost=False,
            confidence=confidence,
        )

    elif query_type == QueryType.PERSONAL:
        plan = QueryPlan(
            query_type=query_type,
            strategy="hybrid",
            top_k=default_top_k,
            use_reranker=False,
            use_vectors=True,
            expanded_queries=[query],
            recency_boost=True,           # Prefer recent personal data
            confidence=confidence,
        )

    elif query_type == QueryType.CODE:
        plan = QueryPlan(
            query_type=query_type,
            strategy="bm25_only",
            top_k=default_top_k,
            use_reranker=False,
            use_vectors=False,            # Code has precise keywords
            expanded_queries=[query],
            recency_boost=False,
            confidence=confidence,
        )

    else:  # EXPLORATORY or UNKNOWN
        plan = QueryPlan(
            query_type=query_type,
            strategy="hybrid",
            top_k=default_top_k,
            use_reranker=False,
            use_vectors=True,
            expanded_queries=[query],
            recency_boost=True,
            confidence=0.3,
        )

    return plan


def _temporal_expansions(query: str) -> List[str]:
    """Expand temporal queries with time-related terms."""
    expansions = [query]

    # Add general recency terms
    expansions.append(query + " recent latest new")

    # Detect specific time references
    import re
    days_match = re.search(r'(\d+)\s+(days?|weeks?|months?)\s+ago', query.lower())
    if days_match:
        expansions.append(query + " activity work discussion")

    return expansions


# ─── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Query planner")
    parser.add_argument("query", help="Query to analyze")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    plan = plan_query(args.query, args.top_k)
    print(f"\nQuery: {args.query}")
    print(f"Type: {plan.query_type.value}")
    print(f"Confidence: {plan.confidence:.2f}")
    print(f"Strategy: {plan.strategy}")
    print(f"Top K: {plan.top_k}")
    print(f"Use Reranker: {plan.use_reranker}")
    print(f"Use Vectors: {plan.use_vectors}")
    print(f"Recency Boost: {plan.recency_boost}")
    print(f"Expanded Queries: {plan.expanded_queries}")
