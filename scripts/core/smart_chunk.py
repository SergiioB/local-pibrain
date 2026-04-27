#!/usr/bin/env python3
"""
Advanced chunking strategies for LocalBrain.

The video highlights that chunking strategy is critical RAG infrastructure.
Different content types need different approaches:
  - Fixed-size: simple, predictable
  - Sliding window: overlap for context continuity
  - Recursive: respects document structure (paragraphs → sentences → chars)
  - Semantic: groups by topical coherence
  - Conversation-aware: splits on speaker turns for chat data

Each strategy produces chunks with provenance metadata.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class ChunkStrategy(Enum):
    FIXED = "fixed"
    SLIDING = "sliding"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    CONVERSATION = "conversation"


@dataclass
class Chunk:
    """A single chunk with provenance."""
    text: str
    index: int
    start: int
    end: int
    strategy: str
    hash: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.hash:
            self.hash = hashlib.sha256(self.text.encode()).hexdigest()


# ─── Strategy 1: Fixed-size ─────────────────────────────────────────────

def chunk_fixed(text: str, chunk_size: int = 512,
                min_size: int = 50) -> List[Chunk]:
    """Simple fixed-size chunking. Breaks at chunk_size boundaries.

    Fast and predictable. Good for uniform content like logs.
    """
    if not text or len(text.strip()) < min_size:
        return []

    if len(text) <= chunk_size:
        return [Chunk(text=text.strip(), index=0, start=0,
                       end=len(text), strategy=ChunkStrategy.FIXED.value)]

    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()

        if len(chunk_text) >= min_size:
            chunks.append(Chunk(
                text=chunk_text,
                index=index,
                start=start,
                end=end,
                strategy=ChunkStrategy.FIXED.value,
                metadata={"size": len(chunk_text)},
            ))
            index += 1
        start = end

    return chunks


# ─── Strategy 2: Sliding window ─────────────────────────────────────────

def chunk_sliding(text: str, chunk_size: int = 512,
                  overlap: int = 64, min_size: int = 50) -> List[Chunk]:
    """Sliding window with overlap for context continuity.

    Overlap ensures no information is lost at boundaries.
    Good for prose, articles, documentation.
    """
    if not text or len(text.strip()) < min_size:
        return []

    if len(text) <= chunk_size:
        return [Chunk(text=text.strip(), index=0, start=0,
                       end=len(text), strategy=ChunkStrategy.SLIDING.value)]

    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try to break at sentence boundary within last 20% of chunk
        if end < len(text):
            search_start = start + int(chunk_size * 0.8)
            for bp in ['\n\n', '. ', '。', '\n', '! ', '? ']:
                pos = text.rfind(bp, search_start, end)
                if pos > search_start:
                    end = pos + len(bp)
                    break

        chunk_text = text[start:end].strip()

        if len(chunk_text) >= min_size:
            chunks.append(Chunk(
                text=chunk_text,
                index=index,
                start=start,
                end=end,
                strategy=ChunkStrategy.SLIDING.value,
                metadata={"size": len(chunk_text), "overlap": overlap},
            ))
            index += 1

        start = max(end - overlap, start + 1) if end < len(text) else end

    return chunks


# ─── Strategy 3: Recursive structure-aware ──────────────────────────────

def chunk_recursive(text: str, max_size: int = 512,
                    min_size: int = 50) -> List[Chunk]:
    """Structure-aware recursive chunking.

    Tries to split at natural boundaries in order of preference:
      1. Double newline (paragraphs)
      2. Single newline (lines)
      3. Sentences (. ! ?)
      4. Character-level fallback

    Preserves document structure — best for general content.
    """
    if not text or len(text.strip()) < min_size:
        return []

    if len(text) <= max_size:
        return [Chunk(text=text.strip(), index=0, start=0,
                       end=len(text), strategy=ChunkStrategy.RECURSIVE.value)]

    chunks = []
    index = 0

    def _split_recursive(text_block: str, start_offset: int):
        nonlocal index

        if len(text_block) <= max_size:
            if len(text_block.strip()) >= min_size:
                chunks.append(Chunk(
                    text=text_block.strip(),
                    index=index,
                    start=start_offset,
                    end=start_offset + len(text_block),
                    strategy=ChunkStrategy.RECURSIVE.value,
                ))
                index += 1
            return

        # Try splitting at natural boundaries
        separators = ['\n\n', '\n', '. ', '。', '! ', '? ', '; ', ' ']

        for sep in separators:
            parts = text_block.split(sep)
            if len(parts) < 2:
                continue

            # Try to group parts into chunks of max_size
            current = ""
            current_start = start_offset
            for part in parts:
                candidate = current + sep + part if current else part
                if len(candidate) <= max_size:
                    current = candidate
                else:
                    if current and len(current.strip()) >= min_size:
                        chunks.append(Chunk(
                            text=current.strip(),
                            index=index,
                            start=current_start,
                            end=current_start + len(current),
                            strategy=ChunkStrategy.RECURSIVE.value,
                        ))
                        index += 1
                    current = part
                    current_start = start_offset + text_block.find(part, len(current) if current else 0)

            if current and len(current.strip()) >= min_size:
                chunks.append(Chunk(
                    text=current.strip(),
                    index=index,
                    start=current_start,
                    end=current_start + len(current),
                    strategy=ChunkStrategy.RECURSIVE.value,
                ))
                index += 1
            return

        # Fallback: character-level split
        for i in range(0, len(text_block), max_size):
            piece = text_block[i:i + max_size].strip()
            if len(piece) >= min_size:
                chunks.append(Chunk(
                    text=piece,
                    index=index,
                    start=start_offset + i,
                    end=start_offset + i + len(piece),
                    strategy=ChunkStrategy.RECURSIVE.value,
                    metadata={"fallback": "character_split"},
                ))
                index += 1

    _split_recursive(text, 0)

    # Re-index
    for i, c in enumerate(chunks):
        c.index = i

    return chunks


# ─── Strategy 4: Semantic grouping ──────────────────────────────────────

def chunk_semantic(text: str, max_size: int = 512,
                   min_size: int = 50) -> List[Chunk]:
    """Group text by topical coherence.

    Uses simple heuristics (heading detection, paragraph grouping)
    to keep related content together. No ML required.
    """
    if not text or len(text.strip()) < min_size:
        return []

    # Split into paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        return []

    chunks = []
    current_text = ""
    current_start = 0
    index = 0

    for para in paragraphs:
        para_start = text.find(para, current_start)

        # Check if this paragraph is a heading (starts with # or is short + ends with :)
        is_heading = bool(re.match(r'^#{1,6}\s', para)) or (
            len(para) < 80 and para.endswith(':')
        )

        candidate = current_text + '\n\n' + para if current_text else para

        if len(candidate) <= max_size:
            current_text = candidate
        else:
            # Save current chunk
            if current_text and len(current_text.strip()) >= min_size:
                chunks.append(Chunk(
                    text=current_text.strip(),
                    index=index,
                    start=current_start,
                    end=current_start + len(current_text),
                    strategy=ChunkStrategy.SEMANTIC.value,
                    metadata={"paragraph_count": current_text.count('\n\n') + 1},
                ))
                index += 1

            # Start new chunk with heading context
            if is_heading and len(para) < max_size:
                current_text = para
                current_start = para_start
            else:
                current_text = para
                current_start = para_start

    # Don't forget the last chunk
    if current_text and len(current_text.strip()) >= min_size:
        chunks.append(Chunk(
            text=current_text.strip(),
            index=index,
            start=current_start,
            end=current_start + len(current_text),
            strategy=ChunkStrategy.SEMANTIC.value,
            metadata={"paragraph_count": current_text.count('\n\n') + 1},
        ))

    return chunks


# ─── Strategy 5: Conversation-aware ─────────────────────────────────────

def chunk_conversation(messages: List[dict],
                       max_turns: int = 10,
                       max_chars: int = 2000) -> List[Chunk]:
    """Chunk conversation data respecting speaker turns.

    Each chunk contains complete turns. Groups related exchanges.
    Best for AI chat sessions (Pi, Claude, Codex, etc.).
    """
    if not messages:
        return []

    chunks = []
    current_turns = []
    current_text = ""
    current_start = 0
    index = 0
    char_offset = 0

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        if not content:
            continue

        turn_text = f"{role}: {content}\n"
        candidate = current_text + turn_text

        if len(candidate) > max_chars and current_turns:
            # Save current chunk
            chunks.append(Chunk(
                text=current_text.strip(),
                index=index,
                start=current_start,
                end=char_offset,
                strategy=ChunkStrategy.CONVERSATION.value,
                metadata={
                    "turn_count": len(current_turns),
                    "roles": list(set(m.get("role", "") for m in current_turns)),
                },
            ))
            index += 1
            current_turns = []
            current_text = turn_text
            current_start = char_offset
        else:
            current_text = candidate

        current_turns.append(msg)
        char_offset += len(turn_text)

    # Last chunk
    if current_text and current_text.strip():
        chunks.append(Chunk(
            text=current_text.strip(),
            index=index,
            start=current_start,
            end=char_offset,
            strategy=ChunkStrategy.CONVERSATION.value,
            metadata={
                "turn_count": len(current_turns),
                "roles": list(set(m.get("role", "") for m in current_turns)),
            },
        ))

    return chunks


# ─── Auto-strategy selection ─────────────────────────────────────────────

def auto_chunk(text: str, source_type: str = "unknown",
               max_size: int = 512, min_size: int = 50,
               overlap: int = 64) -> List[Chunk]:
    """Automatically select the best chunking strategy based on content type.

    Content type heuristics:
      - Conversation data → conversation-aware
      - Structured docs (markdown with headers) → semantic
      - Code/log files → fixed-size
      - General text → recursive (structure-aware)
    """
    if not text or len(text.strip()) < min_size:
        return []

    # Detect conversation format
    if source_type in ("pi", "claude", "codex", "gemini", "droid", "factory",
                       "qwen_code", "opencode", "cursor"):
        # Check if text has conversation turns
        if re.search(r'(user|assistant|human|ai):\s', text[:500], re.IGNORECASE):
            messages = []
            for match in re.finditer(
                r'(user|assistant|human|ai):\s(.*?)(?=\n(?:user|assistant|human|ai):|$)',
                text, re.DOTALL | re.IGNORECASE
            ):
                messages.append({"role": match.group(1).lower(), "content": match.group(2).strip()})
            if len(messages) >= 2:
                return chunk_conversation(messages, max_chars=max_size)

    # Detect markdown/structured docs
    heading_count = len(re.findall(r'^#{1,6}\s', text, re.MULTILINE))
    if heading_count >= 3:
        return chunk_semantic(text, max_size, min_size)

    # Detect code-heavy content
    code_blocks = len(re.findall(r'```', text))
    if code_blocks >= 4 or source_type == "code":
        return chunk_fixed(text, max_size, min_size)

    # Default: recursive (structure-aware)
    return chunk_recursive(text, max_size, min_size)


# ─── Batch chunking for pipeline ─────────────────────────────────────────

def chunk_record(text: str, source_type: str = "unknown",
                 strategy: str = "auto",
                 chunk_size: int = 512,
                 overlap: int = 64) -> List[dict]:
    """Chunk a single record and return dict results suitable for DB insertion."""
    if strategy == "auto":
        chunks = auto_chunk(text, source_type, chunk_size, min_size=50, overlap=overlap)
    elif strategy == "fixed":
        chunks = chunk_fixed(text, chunk_size)
    elif strategy == "sliding":
        chunks = chunk_sliding(text, chunk_size, overlap)
    elif strategy == "recursive":
        chunks = chunk_recursive(text, chunk_size)
    elif strategy == "semantic":
        chunks = chunk_semantic(text, chunk_size)
    else:
        chunks = chunk_sliding(text, chunk_size, overlap)

    return [
        {
            "text": c.text,
            "index": c.index,
            "start": c.start,
            "end": c.end,
            "hash": c.hash,
            "strategy": c.strategy,
            "metadata": json.dumps(c.metadata),
        }
        for c in chunks
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Smart chunking strategies")
    parser.add_argument("--strategy", default="auto",
                        choices=["auto", "fixed", "sliding", "recursive", "semantic"])
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--source", default="unknown",
                        help="Source type hint (pi, claude, code, etc.)")
    parser.add_argument("--text", help="Text to chunk")
    parser.add_argument("--file", help="File to chunk")
    args = parser.parse_args()

    text = args.text or ""
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8", errors="replace")

    if not text:
        print("Provide --text or --file")
        exit(1)

    chunks = chunk_record(text, args.source, args.strategy, args.size, args.overlap)

    print(f"\nStrategy: {args.strategy} | Input: {len(text)} chars | Chunks: {len(chunks)}\n")
    for c in chunks:
        print(f"[{c['index']}] strategy={c['strategy']} "
              f"chars={len(c['text'])} hash={c['hash'][:12]}...")
        print(f"    {c['text'][:120]}...")
        print()
