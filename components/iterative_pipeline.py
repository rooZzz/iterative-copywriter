import json
import re
from openai import OpenAI
from langflow.custom import Component
from langflow.io import StrInput, IntInput, SecretStrInput, MultilineInput, Output
from langflow.schema.message import Message


def _generate_copies(client, model, topic, persona, tone, copy_structure,
                     length_constraints, keywords_required, keywords_excluded, batch_size):
    length_desc = ", ".join(
        f"{k}: {v['min']}-{v['max']} characters"
        for k, v in length_constraints.items()
    )

    if copy_structure == "header":
        structure_desc = 'Each copy must be a JSON object with a "header" key.'
    else:
        structure_desc = (
            'Each copy must be a JSON object with "header" and "subheader" keys '
            "forming a coherent message."
        )

    parts = [
        "You are an expert marketing copywriter.",
        f"Generate exactly {batch_size} different marketing copies.",
        "",
        f"Topic: {topic}",
        f"Target audience: {persona}",
        f"Tone: {tone}",
        "",
        "Constraints:",
        f"- {structure_desc}",
        f"- Length: {length_desc}",
    ]

    if keywords_required:
        parts.append(f"- Must include keywords: {', '.join(keywords_required)}")
    if keywords_excluded:
        parts.append(f"- Must NOT use these words: {', '.join(keywords_excluded)}")

    parts.append("")
    parts.append(
        'Return a JSON object with a "copies" key containing an array of copy objects. '
        "Be creative and diverse."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "\n".join(parts)},
            {"role": "user", "content": f"Generate {batch_size} copies about: {topic}"},
        ],
        temperature=0.9,
        response_format={"type": "json_object"},
    )

    raw = json.loads(response.choices[0].message.content)
    return raw.get("copies", [raw] if isinstance(raw, dict) else raw)


def _format_copy(copy_dict, copy_structure):
    formatted = {}
    for key in ("header", "subheader"):
        if key not in copy_dict:
            continue
        text = copy_dict[key]
        text = text.replace(" and ", " & ")
        text = re.sub(r",(\s+)(and|or)\s", r"\1\2 ", text)
        if copy_structure == "header" or key == "header":
            text = text.rstrip(".")
        formatted[key] = text.strip()
    return formatted


def _check_length(copy_dict, length_constraints):
    for key, bounds in length_constraints.items():
        if key not in copy_dict:
            return False, f"Missing '{key}' in copy"
        length = len(copy_dict[key])
        if length < bounds["min"] or length > bounds["max"]:
            return False, (
                f"Length of '{key}' is {length} chars, "
                f"needs {bounds['min']}-{bounds['max']}"
            )
    return True, "Length OK"


def _check_keywords(copy_dict, required, excluded):
    full_text = " ".join(copy_dict.values()).lower()
    for kw in required:
        if kw.lower() not in full_text:
            return False, f"Required keyword '{kw}' not found"
    for kw in excluded:
        if kw.lower() in full_text:
            return False, f"Excluded word '{kw}' found"
    return True, "Keywords OK"


def _check_tone(client, model, copy_dict, tone):
    copy_text = " | ".join(f"{k}: {v}" for k, v in copy_dict.items())

    prompt = "\n".join([
        "You are a copy editor evaluating tone of voice.",
        "",
        f"Required tone: {tone}",
        "",
        f"Copy to evaluate:\n{copy_text}",
        "",
        "Think step by step:",
        "1. Analyze the words and phrases used",
        "2. Determine the tone conveyed",
        "3. Compare against the required tone",
        "4. Decide if the copy meets the requirements",
        "",
        'Return JSON: {"pass": true/false, "reasoning": "...", "feedback": "..."}',
    ])

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    return result.get("pass", False), result.get("feedback", "Tone check failed")


def _evaluate_copy(client, model, copy_dict, constraints):
    passed, feedback = _check_length(copy_dict, constraints.get("length", {}))
    if not passed:
        return False, feedback

    passed, feedback = _check_keywords(
        copy_dict,
        constraints.get("keywords_required", []),
        constraints.get("keywords_excluded", []),
    )
    if not passed:
        return False, feedback

    tone = constraints.get("tone", "")
    if tone:
        passed, feedback = _check_tone(client, model, copy_dict, tone)
        if not passed:
            return False, feedback

    return True, "All checks passed"


def _refine_copy(client, model, copy_dict, feedback, constraints):
    constraint_lines = []
    if "topic" in constraints:
        constraint_lines.append(f"Topic: {constraints['topic']}")
    if "tone" in constraints:
        constraint_lines.append(f"Tone: {constraints['tone']}")
    if "length" in constraints:
        for key, bounds in constraints["length"].items():
            constraint_lines.append(
                f"Length of {key}: {bounds['min']}-{bounds['max']} characters"
            )
    if constraints.get("keywords_required"):
        constraint_lines.append(
            f"Required keywords: {', '.join(constraints['keywords_required'])}"
        )

    prompt = "\n".join([
        "You are an expert marketing copywriter. Refine this copy based on feedback.",
        "",
        f"Original copy:\n{json.dumps(copy_dict, indent=2)}",
        "",
        f"Feedback:\n{feedback}",
        "",
        "Constraints:",
        *[f"- {line}" for line in constraint_lines],
        "",
        "Return ONLY a JSON object with the same keys as the original.",
    ])

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)


class IterativeRefinementPipeline(Component):
    display_name = "Iterative Refinement Pipeline"
    description = (
        "Full copy generation pipeline: generate -> format -> evaluate -> refine loop. "
        "Implements the iterative refinement framework from the paper."
    )
    icon = "repeat"

    inputs = [
        SecretStrInput(name="api_key", display_name="OpenAI API Key", required=True),
        StrInput(name="model", display_name="Model", value="gpt-4o-mini"),
        StrInput(
            name="topic",
            display_name="Topic",
            required=True,
            info="The service or product to write copy about",
        ),
        StrInput(name="persona", display_name="Target Persona", value="general audience"),
        StrInput(name="tone", display_name="Tone of Voice", value="confident, friendly, concise"),
        StrInput(
            name="copy_structure",
            display_name="Copy Structure",
            value="header",
            info="Either 'header' or 'header_subheader'",
        ),
        MultilineInput(
            name="length_constraints",
            display_name="Length Constraints JSON",
            value='{"header": {"min": 30, "max": 60}}',
        ),
        MultilineInput(
            name="keywords_required",
            display_name="Required Keywords JSON",
            value="[]",
        ),
        MultilineInput(
            name="keywords_excluded",
            display_name="Excluded Keywords JSON",
            value="[]",
        ),
        IntInput(name="batch_size", display_name="Batch Size", value=5),
        IntInput(
            name="max_refinements",
            display_name="Max Refinement Attempts",
            value=2,
            info="Maximum number of refine attempts per copy (paper recommends 1-2)",
        ),
    ]

    outputs = [
        Output(display_name="Pipeline Results", name="results", method="run_pipeline"),
    ]

    def run_pipeline(self) -> Message:
        client = OpenAI(api_key=self.api_key)

        length_constraints = json.loads(self.length_constraints)
        required_kw = json.loads(self.keywords_required)
        excluded_kw = json.loads(self.keywords_excluded)

        constraints = {
            "topic": self.topic,
            "tone": self.tone,
            "length": length_constraints,
            "keywords_required": required_kw,
            "keywords_excluded": excluded_kw,
        }

        raw_copies = _generate_copies(
            client, self.model, self.topic, self.persona, self.tone,
            self.copy_structure, length_constraints, required_kw, excluded_kw,
            self.batch_size,
        )

        accepted = []
        rejected = []
        log = []

        for i, raw_copy in enumerate(raw_copies):
            formatted = _format_copy(raw_copy, self.copy_structure)
            copy_log = {"index": i, "original": raw_copy, "attempts": []}
            passed = False

            for attempt in range(self.max_refinements + 1):
                success, feedback = _evaluate_copy(
                    client, self.model, formatted, constraints
                )

                copy_log["attempts"].append({
                    "attempt": attempt,
                    "copy": dict(formatted),
                    "passed": success,
                    "feedback": feedback,
                })

                if success:
                    accepted.append(formatted)
                    passed = True
                    break

                if attempt < self.max_refinements:
                    refined = _refine_copy(
                        client, self.model, formatted, feedback, constraints
                    )
                    formatted = _format_copy(refined, self.copy_structure)

            if not passed:
                rejected.append(formatted)

            log.append(copy_log)

        total = len(raw_copies)
        success_rate = (len(accepted) / total * 100) if total > 0 else 0.0

        result = {
            "summary": {
                "total_generated": total,
                "accepted": len(accepted),
                "rejected": len(rejected),
                "success_rate": round(success_rate, 2),
            },
            "accepted_copies": accepted,
            "rejected_copies": rejected,
            "detailed_log": log,
        }

        return Message(text=json.dumps(result, indent=2))
