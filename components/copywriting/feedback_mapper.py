import json
import re

from tone_guidelines import format_guidelines_for_prompt


_TEMPLATES = {
    "length": (
        "The {component} is {length} characters but must be between "
        "{min} and {max}. {action} to fit within the required range "
        "while preserving the core message."
    ),
    "keywords_missing": (
        "The required keyword '{keyword}' is missing. "
        "Incorporate it naturally into the copy without forcing it."
    ),
    "keywords_excluded": (
        "The excluded word '{keyword}' appears in the copy. "
        "Replace it with an alternative that conveys the same meaning."
    ),
    "punctuation": (
        "Punctuation issue: {detail}. "
        "Adjust the punctuation to comply with brand guidelines."
    ),
    "off_brand": (
        "The word '{word}' is off-brand. "
        "Replace it with language that aligns with the brand voice."
    ),
    "british_english": (
        "{detail} "
        "Use British English spelling throughout."
    ),
    "coherence": (
        "The header and subheader lack coherence. "
        "{detail} Revise so they form a connected, complementary message."
    ),
    "topic_relevance": (
        "The copy does not clearly convey the intended topic ({topic}). "
        "{detail} Refocus the message on the value proposition for {persona}."
    ),
    "lexical_ordering": (
        "The preferred word ordering is not respected. "
        "{detail} Rearrange phrasing to follow the specified ordering rules."
    ),
}


def _parse_length_feedback(raw_feedback):
    match = re.search(
        r"Length of '(\w+)' is (\d+) characters.*?between (\d+) and (\d+)",
        raw_feedback,
    )
    if not match:
        match = re.search(
            r"Length of '(\w+)' is (\d+) chars.*?needs (\d+)-(\d+)",
            raw_feedback,
        )
    if match:
        component = match.group(1)
        length = int(match.group(2))
        min_len = int(match.group(3))
        max_len = int(match.group(4))
        action = "Shorten it" if length > max_len else "Expand it"
        return _TEMPLATES["length"].format(
            component=component,
            length=length,
            min=min_len,
            max=max_len,
            action=action,
        )
    return None


def _parse_keyword_feedback(raw_feedback):
    missing = re.search(r"Required keyword '(.+?)' not found", raw_feedback)
    if missing:
        return _TEMPLATES["keywords_missing"].format(keyword=missing.group(1))

    excluded = re.search(r"Excluded word '(.+?)' found", raw_feedback)
    if excluded:
        return _TEMPLATES["keywords_excluded"].format(keyword=excluded.group(1))

    return None


def _parse_off_brand_feedback(raw_feedback):
    match = re.search(r"Off-brand word '(.+?)' found", raw_feedback)
    if match:
        return _TEMPLATES["off_brand"].format(word=match.group(1))
    return None


def _build_tone_instruction(raw_feedback, tone_guidelines):
    parts = [
        "The tone does not match the brand voice guidelines.",
        raw_feedback,
        "",
        "Rewrite to match these tone guidelines:",
        format_guidelines_for_prompt(tone_guidelines),
    ]
    return "\n".join(parts)


def map_feedback_to_instruction(evaluator_name, raw_feedback, context):
    topic = context.get("topic", "")
    persona = context.get("persona", "general audience")
    tone_guidelines = context.get("tone_guidelines", {})

    if evaluator_name == "length":
        result = _parse_length_feedback(raw_feedback)
        if result:
            return result

    if evaluator_name == "keywords":
        result = _parse_keyword_feedback(raw_feedback)
        if result:
            return result

    if evaluator_name == "off_brand":
        result = _parse_off_brand_feedback(raw_feedback)
        if result:
            return result

    if evaluator_name == "british_english":
        return _TEMPLATES["british_english"].format(detail=raw_feedback)

    if evaluator_name == "punctuation":
        return _TEMPLATES["punctuation"].format(detail=raw_feedback)

    if evaluator_name == "tone":
        return _build_tone_instruction(raw_feedback, tone_guidelines)

    if evaluator_name == "coherence":
        return _TEMPLATES["coherence"].format(detail=raw_feedback)

    if evaluator_name == "topic_relevance":
        return _TEMPLATES["topic_relevance"].format(
            topic=topic, detail=raw_feedback, persona=persona,
        )

    if evaluator_name == "lexical_ordering":
        return _TEMPLATES["lexical_ordering"].format(detail=raw_feedback)

    return raw_feedback


def build_refiner_prompt(copy_dict, evaluator_name, raw_feedback, constraints):
    context = {
        "topic": constraints.get("topic", ""),
        "persona": constraints.get("persona", "general audience"),
        "tone_guidelines": constraints.get("tone_guidelines", {}),
    }

    instruction = map_feedback_to_instruction(evaluator_name, raw_feedback, context)

    tone_guidelines = constraints.get("tone_guidelines", {})
    guidelines_block = format_guidelines_for_prompt(tone_guidelines) if tone_guidelines else ""

    constraint_lines = []
    if context["topic"]:
        constraint_lines.append(f"Topic: {context['topic']}")
    if context["persona"]:
        constraint_lines.append(f"Target audience: {context['persona']}")
    if guidelines_block:
        constraint_lines.append(f"Tone of voice guidelines:\n{guidelines_block}")
    if "length" in constraints:
        for key, bounds in constraints["length"].items():
            constraint_lines.append(
                f"Length of {key}: {bounds['min']}-{bounds['max']} characters"
            )
    if constraints.get("keywords_required"):
        constraint_lines.append(
            f"Required keywords: {', '.join(constraints['keywords_required'])}"
        )
    if constraints.get("keywords_excluded"):
        constraint_lines.append(
            f"Excluded words: {', '.join(constraints['keywords_excluded'])}"
        )

    prompt = "\n".join([
        "You are an expert marketing copywriter writing for a British audience.",
        "All output MUST use British English spelling and conventions.",
        "Revise the copy below according to the instruction.",
        "",
        f"Original copy:\n{json.dumps(copy_dict, indent=2)}",
        "",
        f"Instruction:\n{instruction}",
        "",
        "Constraints to maintain:",
        *[f"- {line}" for line in constraint_lines],
        "",
        "Return ONLY a JSON object with the same keys as the original copy.",
    ])

    return prompt
