#!/usr/bin/env python3
"""
Extract conversations from Qwen Code sessions.
Scans home/.qwen/projects/*/chats/*.jsonl for conversation files.
"""
import json
import sys
import os
from pathlib import Path
from typing import Iterator


def extract_text_from_message(entry) -> str:
    """Extract text from Qwen Code message format."""
    msg = entry.get("message", {})
    parts = msg.get("parts", [])
    texts = []
    for part in parts:
        if isinstance(part, dict):
            t = part.get("text", "")
            if t:
                # Skip system reminders
                if "<system-reminder>" in t:
                    end = t.rfind("</system-reminder>")
                    if end > 0:
                        t = t[end + len("</system-reminder>"):]
                texts.append(t.strip())
        elif isinstance(part, str):
            texts.append(part.strip())
    return "\n".join(t for t in texts if t)


def extract_qwen_session(jsonl_path: Path) -> Iterator[dict]:
    """Extract a single Qwen Code session."""
    session_id = jsonl_path.stem
    project_dir = jsonl_path.parent.parent
    project_name = project_dir.name

    # Decode URL-encoded project path
    display_project = project_name.replace("-", "/").strip("/")

    messages = []
    tool_calls_count = 0

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

                # Only process user and assistant messages
                if entry_type not in ("user", "assistant"):
                    continue

                msg = entry.get("message", {})
                role = msg.get("role", entry_type)

                text = extract_text_from_message(entry)
                if not text.strip():
                    continue

                messages.append({
                    "role": role,
                    "content": text,
                })

                # Count tool usage
                if entry_type == "assistant":
                    tool_results = msg.get("toolResults", [])
                    if tool_results:
                        tool_calls_count += len(tool_results)

    except Exception as e:
        print(f"  Error reading {jsonl_path.name}: {e}", file=sys.stderr)
        return

    if not messages:
        return

    # Build title from first user message
    title = session_id
    for m in messages:
        if m["role"] == "user":
            title = m["content"][:120].strip()
            break

    # Determine category
    total_text = " ".join(m["content"] for m in messages).lower()
    category = "chat"
    if any(kw in total_text for kw in ["def ", "class ", "import ", "function ", "code", "script", "python"]):
        category = "code"
    elif any(kw in total_text for kw in ["tool", "shell", "command", "execute", "run"]):
        category = "tool"
    elif tool_calls_count > 0:
        category = "tool"
    elif any(kw in total_text for kw in ["analysis", "think", "reason", "plan"]):
        category = "reasoning"

    content_text = "\n\n".join(f'{m["role"]}: {m["content"]}' for m in messages)

    record = {
        "external_id": session_id,
        "record_type": "conversation",
        "title": title,
        "created_at": "",
        "category": category,
        "tags": ["qwen_code"],
        "messages": messages,
        "metadata": {
            "source": "qwen_code",
            "project": display_project,
            "session_id": session_id,
            "tools_used_count": tool_calls_count,
            "num_messages": len(messages),
            "content_text": content_text,
            "content_preview": content_text[:2000],
        },
    }

    yield record


def discover_qwen_sessions(qwen_dir: Path = None) -> list[Path]:
    """Find all Qwen Code session JSONL files."""
    if qwen_dir is None:
        qwen_dir = Path(os.path.expanduser("~")) / ".qwen" / "projects"

    if not qwen_dir.exists():
        return []

    jsonl_files = []
    for root, dirs, files in os.walk(str(qwen_dir)):
        root_path = Path(root)
        if root_path.name == "chats":
            for f in files:
                if f.endswith(".jsonl"):
                    jsonl_files.append(root_path / f)

    return sorted(jsonl_files)


def export_qwen_to_jsonl(qwen_dir: Path = None, output_path: Path = None):
    """Export all Qwen Code sessions to a single JSONL file."""
    sessions = discover_qwen_sessions(qwen_dir)
    if not sessions:
        print(f"No Qwen Code sessions found")
        return 0

    if output_path is None:
        output_path = Path("exports/qwen_code_conversations.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    total_messages = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for jsonl_path in sessions:
            for record in extract_qwen_session(jsonl_path):
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                total_messages += record["metadata"]["num_messages"]
                if count % 50 == 0:
                    print(f"  Exported {count} sessions...")

    print(f"Exported {count} Qwen Code sessions ({total_messages} messages) to {output_path}")
    return count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export Qwen Code sessions")
    parser.add_argument("--qwen-dir", type=Path, default=None,
                        help="Path to .qwen/projects directory")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSONL file path")
    args = parser.parse_args()

    count = export_qwen_to_jsonl(args.qwen_dir, args.output)
    if count == 0:
        sys.exit(1)
