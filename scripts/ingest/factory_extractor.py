#!/usr/bin/env python3
"""
Extract conversations from .factory sessions (DROID/Factory AI tool).
Scans home/.factory/sessions/ for JSONL conversation files.
"""
import json
import sys
import os
import re
from pathlib import Path
from typing import Iterator
from datetime import datetime


def extract_text_content(message) -> str:
    """Extract plain text from a .factory message content array."""
    content = message.get("content", [])
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")
                if text:
                    # Skip system reminders for cleaner extraction
                    if "<system-reminder>" in text and len(text) > 500:
                        # Keep only user's actual message
                        # System reminders are at the start
                        end_reminder = text.rfind("</system-reminder>")
                        if end_reminder > 0:
                            text = text[end_reminder + len("</system-reminder>"):]
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(p for p in parts if p.strip())
    return str(content)


def get_model_from_settings(settings_path: Path) -> str:
    """Extract model name from session settings file."""
    try:
        if settings_path.exists():
            with open(settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("model", "unknown")
    except Exception:
        pass
    return "unknown"


def extract_factory_session(jsonl_path: Path) -> Iterator[dict]:
    """Extract a single .factory session as a conversation record."""
    settings_path = jsonl_path.with_suffix(".settings.json")
    model = get_model_from_settings(settings_path)

    session_id = jsonl_path.stem
    session_title = ""
    session_owner = ""
    session_cwd = ""
    messages = []

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry_type = entry.get("type", "")

                if entry_type == "session_start":
                    session_title = entry.get("title") or entry.get("sessionTitle", "")
                    session_owner = entry.get("owner", "")
                    session_cwd = entry.get("cwd", "")

                elif entry_type == "message":
                    msg = entry.get("message", {})
                    role = msg.get("role", "unknown")
                    # Skip visibility-restricted messages
                    visibility = msg.get("visibility", "")
                    if visibility == "user_only" and role != "user":
                        continue
                    text = extract_text_content(msg)
                    if text.strip():
                        messages.append({
                            "role": role,
                            "content": text,
                            "timestamp": entry.get("timestamp", ""),
                        })
    except Exception as e:
        print(f"  Error reading {jsonl_path.name}: {e}", file=sys.stderr)
        return

    if not messages:
        return

    # Build title from session info
    title = session_title
    if not title and messages:
        # Use first user message as title
        for m in messages:
            if m["role"] == "user":
                title = m["content"][:100].strip()
                break
    if not title:
        title = session_id

    # Determine category
    total_text = " ".join(m["content"] for m in messages).lower()
    category = "chat"
    if any(kw in total_text for kw in ["def ", "class ", "import ", "function ", "code"]):
        category = "code"
    elif any(kw in total_text for kw in ["tool", "execute", "script", "command"]):
        category = "tool"
    elif any(kw in total_text for kw in ["analysis", "think", "reason"]):
        category = "reasoning"

    content_text = "\n\n".join(f'{m["role"]}: {m["content"]}' for m in messages)

    # Extract path context from session folder name
    # e.g., -C-Users-username-projects
    path_context = ""
    parent = jsonl_path.parent.name
    if parent.startswith("-"):
        path_context = parent.replace("-", "/").strip("/")

    record = {
        "external_id": session_id,
        "record_type": "conversation",
        "title": title,
        "created_at": messages[0].get("timestamp", ""),
        "category": category,
        "tags": ["factory", model],
        "messages": messages,
        "metadata": {
            "source": "factory",
            "model": model,
            "owner": session_owner,
            "cwd": session_cwd,
            "path_context": path_context,
            "num_messages": len(messages),
            "content_text": content_text,
            "content_preview": content_text[:2000],
        },
    }

    yield record


def discover_factory_sessions(factory_dir: Path = None) -> list[Path]:
    """Find all .factory session JSONL files."""
    if factory_dir is None:
        # Default: C:\Users\<user>\.factory
        factory_dir = Path.home() / ".factory" / "sessions"

    if not factory_dir.exists():
        return []

    jsonl_files = []
    for root, dirs, files in os.walk(factory_dir):
        for f in files:
            if f.endswith(".jsonl") and not f.endswith(".settings.json"):
                jsonl_files.append(Path(root) / f)

    return sorted(jsonl_files)


def export_factory_to_jsonl(factory_dir: Path = None, output_path: Path = None):
    """Export all .factory sessions to a single JSONL file."""
    sessions = discover_factory_sessions(factory_dir)
    if not sessions:
        print(f"No .factory sessions found in {factory_dir}")
        return 0

    if output_path is None:
        output_path = Path("exports/factory_conversations.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for jsonl_path in sessions:
            for record in extract_factory_session(jsonl_path):
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                if count % 100 == 0:
                    print(f"  Exported {count} sessions...")

    print(f"Exported {count} factory sessions to {output_path}")
    return count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export .factory sessions")
    parser.add_argument("--factory-dir", type=Path, default=None,
                        help="Path to .factory/sessions directory")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSONL file path")
    args = parser.parse_args()

    count = export_factory_to_jsonl(args.factory_dir, args.output)
    if count == 0:
        sys.exit(1)
