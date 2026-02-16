import json
import os
import re
from datetime import datetime, timezone
from langflow.custom import Component
from langflow.io import StrInput, MultilineInput, Output
from langflow.schema.message import Message


def load_feedback_log(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def save_feedback_log(path, log):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)


def format_copies_for_review(copies):
    lines = ["Copies for review:", ""]
    for i, copy in enumerate(copies, 1):
        parts = " | ".join(f"{k}: {v}" for k, v in copy.items())
        lines.append(f"  {i}. {parts}")
    lines.append("")
    lines.append("Reply with feedback per copy, e.g.:")
    lines.append('  1: accept')
    lines.append('  2: reject - too salesy')
    lines.append('  3: reject - does not match brand voice')
    return "\n".join(lines)


def parse_review_feedback(feedback_text, copies):
    entries = []
    for line in feedback_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        match = re.match(r"(\d+)\s*:\s*(accept|reject)(?:\s*-\s*(.+))?", line, re.IGNORECASE)
        if not match:
            continue

        index = int(match.group(1))
        decision = match.group(2).lower()
        reason = (match.group(3) or "").strip()

        if 1 <= index <= len(copies):
            entries.append({
                "copy": copies[index - 1],
                "decision": decision,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    return entries


def get_negative_examples(feedback_log, max_examples=5):
    rejected = [
        entry for entry in feedback_log
        if entry.get("decision") == "reject" and entry.get("reason")
    ]
    return rejected[-max_examples:]


class HumanReviewLogger(Component):
    display_name = "Human Review Logger"
    description = "Presents copies for human review, parses feedback, and persists decisions to a log file for future pipeline improvement"
    icon = "clipboard-check"

    inputs = [
        MultilineInput(
            name="copies_json",
            display_name="Accepted Copies JSON",
            required=True,
            info="JSON array of copy objects from the pipeline",
        ),
        MultilineInput(
            name="review_feedback",
            display_name="Review Feedback",
            value="",
            info="Enter feedback per copy, e.g. '1: accept' or '2: reject - too salesy'. Leave blank to just view copies.",
        ),
        StrInput(
            name="feedback_log_path",
            display_name="Feedback Log Path",
            value="/app/langflow/feedback_log.json",
            info="File path for persisting review decisions",
        ),
    ]

    outputs = [
        Output(display_name="Review Summary", name="summary", method="process_review"),
    ]

    def process_review(self) -> Message:
        copies = json.loads(self.copies_json)

        if not self.review_feedback or not self.review_feedback.strip():
            return Message(text=format_copies_for_review(copies))

        entries = parse_review_feedback(self.review_feedback, copies)

        if entries and self.feedback_log_path:
            existing = load_feedback_log(self.feedback_log_path)
            existing.extend(entries)
            save_feedback_log(self.feedback_log_path, existing)

        accepted_count = sum(1 for e in entries if e["decision"] == "accept")
        rejected_count = sum(1 for e in entries if e["decision"] == "reject")

        result = {
            "processed": len(entries),
            "accepted": accepted_count,
            "rejected": rejected_count,
            "log_path": self.feedback_log_path,
            "entries": entries,
        }

        return Message(text=json.dumps(result, indent=2))
