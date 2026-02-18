import json
import re
from openai import OpenAI

from tone_guidelines import format_guidelines_for_prompt


def check_length(copy_dict, length_constraints):
    for key, bounds in length_constraints.items():
        if key not in copy_dict:
            return False, f"Missing '{key}' in copy"

        length = len(copy_dict[key])
        if length < bounds["min"] or length > bounds["max"]:
            return False, (
                f"Length of '{key}' is {length} characters, "
                f"but must be between {bounds['min']} and {bounds['max']}"
            )

    return True, "Length check passed"


def check_keywords(copy_dict, required, excluded):
    full_text = " ".join(copy_dict.values()).lower()

    for kw in required:
        if kw.lower() not in full_text:
            return False, f"Required keyword '{kw}' not found in copy"

    for kw in excluded:
        if kw.lower() in full_text:
            return False, f"Excluded word '{kw}' found in copy"

    return True, "Keyword check passed"


def check_punctuation(copy_dict, rules):
    no_exclamation = rules.get("no_exclamation_marks", False)
    no_ellipsis = rules.get("no_ellipsis", False)
    no_all_caps = rules.get("no_all_caps_words", False)
    max_commas = rules.get("max_commas_per_component", None)

    for key, text in copy_dict.items():
        if no_exclamation and "!" in text:
            return False, f"Exclamation mark found in '{key}': brand guidelines prohibit them"

        if no_ellipsis and "..." in text:
            return False, f"Ellipsis found in '{key}': brand guidelines prohibit them"

        if no_all_caps:
            words = text.split()
            caps_words = [w for w in words if w.isupper() and len(w) > 1]
            if caps_words:
                return False, (
                    f"All-caps words found in '{key}': {', '.join(caps_words)}. "
                    f"Brand guidelines prohibit all-caps words"
                )

        if max_commas is not None:
            comma_count = text.count(",")
            if comma_count > max_commas:
                return False, (
                    f"Too many commas in '{key}': found {comma_count}, "
                    f"maximum allowed is {max_commas}"
                )

    return True, "Punctuation check passed"


def check_off_brand_words(copy_dict, blocklist):
    full_text = " ".join(copy_dict.values()).lower()

    for word in blocklist:
        pattern = r"\b" + re.escape(word.lower()) + r"\b"
        if re.search(pattern, full_text):
            return False, f"Off-brand word '{word}' found in copy"

    return True, "Off-brand words check passed"


AMERICAN_TO_BRITISH = {
    "personalized": "personalised",
    "customize": "customise",
    "customized": "customised",
    "organize": "organise",
    "organized": "organised",
    "organization": "organisation",
    "optimize": "optimise",
    "optimized": "optimised",
    "recognize": "recognise",
    "recognized": "recognised",
    "minimize": "minimise",
    "maximise": "maximise",
    "analyze": "analyse",
    "analyzed": "analysed",
    "utilize": "utilise",
    "utilized": "utilised",
    "prioritize": "prioritise",
    "prioritized": "prioritised",
    "favor": "favour",
    "favorite": "favourite",
    "color": "colour",
    "honor": "honour",
    "behavior": "behaviour",
    "neighbor": "neighbour",
    "catalog": "catalogue",
    "dialog": "dialogue",
    "center": "centre",
    "fiber": "fibre",
    "license": "licence",
    "defense": "defence",
    "offense": "offence",
    "check": "cheque",
    "gray": "grey",
}


def check_british_english(copy_dict):
    full_text = " ".join(copy_dict.values()).lower()

    for american, british in AMERICAN_TO_BRITISH.items():
        pattern = r"\b" + re.escape(american) + r"\b"
        if re.search(pattern, full_text):
            return False, (
                f"American English spelling '{american}' found. "
                f"Use British English: '{british}'"
            )

    return True, "British English check passed"


def check_tone(client, model, copy_dict, tone_guidelines, examples=None):
    copy_text = " | ".join(f"{k}: {v}" for k, v in copy_dict.items())
    guidelines_block = format_guidelines_for_prompt(tone_guidelines)

    on_brand = tone_guidelines.get("examples", {}).get("on_brand", [])
    off_brand = tone_guidelines.get("examples", {}).get("off_brand", [])

    parts = [
        "You are a senior brand copy editor evaluating tone of voice against guidelines.",
        "Be strict but fair. Your job is to catch copy that genuinely misses the brand voice, not to demand perfection.",
        "",
        "Tone of voice guidelines:",
        guidelines_block,
        "",
    ]

    if on_brand:
        parts.append("ON-BRAND EXAMPLES (use these as your reference for what good looks like):")
        for i, ex in enumerate(on_brand, 1):
            parts.append(f"  {i}. {ex}")
        parts.append("")

    if off_brand:
        parts.append("OFF-BRAND EXAMPLES (reject copy that sounds more like these than the on-brand examples):")
        for i, ex in enumerate(off_brand, 1):
            parts.append(f"  {i}. {ex}")
        parts.append("")

    if examples:
        parts.append("Previous evaluation examples for calibration:")
        for i, ex in enumerate(examples, 1):
            parts.append(f"Example {i}: {json.dumps(ex)}")
        parts.append("")

    parts.extend([
        f"Copy to evaluate:\n{copy_text}",
        "",
        "REJECT the copy if ANY of these are true:",
        "- It communicates no specific benefit or action — just vague aspirational language (e.g. 'Unlock your potential', 'Your journey to freedom', 'Take control')",
        "- The sentence structure clearly clashes with the on-brand examples (e.g. flowery and long when examples are short and direct)",
        "- It reads more like the off-brand examples than the on-brand examples",
        "- It is GENERIC: you could replace the product name with any competitor and the copy would still work. Good copy is grounded in specific details.",
        "- It contains words like 'effectively', 'seamlessly', 'effortlessly' that add no meaning",
        "",
        "PASS the copy if:",
        "- It communicates a specific benefit, feature, or action grounded in real product details",
        "- The style, structure, and register are consistent with the on-brand examples",
        "- The formality level matches the guidelines",
        "- It may not be perfect, but it genuinely sounds like it belongs to this brand and refers to something concrete",
        "",
        "Think step by step:",
        "1. What specific claim or benefit does the copy communicate? If the answer is 'nothing specific' or just a vague category like 'better credit', REJECT.",
        "2. Does the style and structure match the on-brand examples?",
        "3. Does it sound more like the on-brand or off-brand examples?",
        "4. Is the formality and emotional register correct?",
        "",
        "CRITICAL: When providing feedback or rewrite suggestions, use ONLY facts and details that appear in the tone guidelines or the copy itself. Do NOT invent statistics, user counts, timeframes, or features. If you mention a number, it must come from the guidelines or the copy being evaluated. Hallucinating facts is strictly forbidden.",
        "",
        'Return a JSON object: {"pass": true/false, "reasoning": "...", "feedback": "..."}',
    ])

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "\n".join(parts)}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    return result.get("pass", False), result.get("feedback", "Tone evaluation failed")


def check_coherence(client, model, copy_dict, examples=None):
    if len(copy_dict) < 2:
        return True, "Coherence check skipped: single-component copy"

    copy_text = "\n".join(f"{k}: {v}" for k, v in copy_dict.items())

    parts = [
        "You are a copy editor evaluating whether a header and subheader work well together as marketing copy.",
        "",
        f"Copy components:\n{copy_text}",
        "",
    ]

    if examples:
        parts.append("Previous evaluation examples for calibration:")
        for i, ex in enumerate(examples, 1):
            parts.append(f"Example {i}: {json.dumps(ex)}")
        parts.append("")

    parts.extend([
        "In good marketing copy, the header and subheader have DIFFERENT but COMPLEMENTARY roles:",
        "- Header: the hook — grabs attention with a key benefit, claim, or action",
        "- Subheader: the detail — adds supporting information, specifics, or a secondary benefit",
        "",
        "PASS if:",
        "- The header and subheader are about the same product/topic (they don't contradict each other)",
        "- The subheader adds NEW information rather than just restating the header",
        "- A reader would understand the overall message when reading both together",
        "- It is completely fine for the header to focus on one benefit (e.g. debt reduction) and the subheader on another (e.g. credit score improvement) — these are complementary, not incoherent",
        "",
        "REJECT only if:",
        "- The header and subheader actively contradict each other",
        "- The subheader is just the header rephrased with no new information",
        "- They seem to be about completely unrelated topics",
        "",
        'Return a JSON object: {"pass": true/false, "reasoning": "...", "feedback": "..."}',
    ])

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "\n".join(parts)}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    return result.get("pass", False), result.get("feedback", "Coherence evaluation failed")


def check_topic_relevance(client, model, copy_dict, topic, persona, examples=None):
    copy_text = " | ".join(f"{k}: {v}" for k, v in copy_dict.items())

    parts = [
        "You are a strict marketing strategist evaluating whether copy actually communicates the intended topic.",
        "Be demanding. The copy must convey the SPECIFIC value proposition, not just vaguely gesture at the category.",
        "",
        f"Intended topic: {topic}",
        f"Target audience: {persona}",
        "",
    ]

    if examples:
        parts.append("Previous evaluation examples for calibration:")
        for i, ex in enumerate(examples, 1):
            parts.append(f"Example {i}: {json.dumps(ex)}")
        parts.append("")

    parts.extend([
        f"Copy to evaluate:\n{copy_text}",
        "",
        "REJECT if:",
        "- The copy only vaguely relates to the topic (e.g. topic is 'CreditFixer debt reduction tool' but copy just says 'better credit')",
        "- The copy could apply to ANY product in the same category, not specifically THIS product/service",
        "- The key differentiators or specific features mentioned in the topic are missing from the copy",
        "- A reader would not understand what the product actually DOES from reading the copy alone",
        "",
        "PASS only if:",
        "- The copy clearly communicates the specific product/service and its key benefit",
        "- A reader unfamiliar with the brand would understand the core value proposition",
        "- The copy addresses the target audience's actual needs",
        "",
        "Think step by step:",
        "1. What specific product, service, or feature does the topic describe?",
        "2. Does the copy communicate that specific thing, or just the general category?",
        "3. Would a reader understand what is being offered and why they should care?",
        "4. Is there enough specificity that this copy could ONLY be about this topic?",
        "",
        "If rejecting, explain exactly what topic details are missing and suggest how to include them.",
        "",
        'Return a JSON object: {"pass": true/false, "reasoning": "...", "feedback": "..."}',
    ])

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "\n".join(parts)}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    return result.get("pass", False), result.get("feedback", "Topic relevance evaluation failed")


def check_lexical_ordering(client, model, copy_dict, preferred_order, examples=None):
    copy_text = " | ".join(f"{k}: {v}" for k, v in copy_dict.items())

    order_desc = "\n".join(
        f"- '{item['before']}' should appear before '{item['after']}'"
        for item in preferred_order
    )

    parts = [
        "You are a copy editor evaluating whether a marketing copy follows preferred word ordering.",
        "",
        f"Preferred ordering rules:\n{order_desc}",
        "",
    ]

    if examples:
        parts.append("Here are examples of previous evaluations for calibration:")
        for i, ex in enumerate(examples, 1):
            parts.append(f"Example {i}: {json.dumps(ex)}")
        parts.append("")

    parts.extend([
        f"Copy to evaluate:\n{copy_text}",
        "",
        "Think step by step:",
        "1. For each ordering rule, locate the relevant phrases in the copy",
        "2. Check if the preferred order is maintained",
        "3. Note any violations",
        "",
        'Return a JSON object: {"pass": true/false, "reasoning": "...", "feedback": "..."}',
    ])

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "\n".join(parts)}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    return result.get("pass", False), result.get("feedback", "Lexical ordering evaluation failed")


EVALUATOR_REGISTRY = {
    "length": {
        "fn": check_length,
        "type": "deterministic",
    },
    "keywords": {
        "fn": check_keywords,
        "type": "deterministic",
    },
    "punctuation": {
        "fn": check_punctuation,
        "type": "deterministic",
    },
    "off_brand": {
        "fn": check_off_brand_words,
        "type": "deterministic",
    },
    "british_english": {
        "fn": check_british_english,
        "type": "deterministic",
    },
    "tone": {
        "fn": check_tone,
        "type": "llm",
    },
    "coherence": {
        "fn": check_coherence,
        "type": "llm",
    },
    "topic_relevance": {
        "fn": check_topic_relevance,
        "type": "llm",
    },
    "lexical_ordering": {
        "fn": check_lexical_ordering,
        "type": "llm",
    },
}

PRESETS = {
    "basic": ["length", "keywords", "british_english"],
    "standard": ["length", "keywords", "british_english", "tone"],
    "full": [
        "length",
        "keywords",
        "british_english",
        "punctuation",
        "off_brand",
        "tone",
        "coherence",
        "topic_relevance",
        "lexical_ordering",
    ],
}


def run_evaluator(name, client, model, copy_dict, constraints, eval_examples=None):
    if name not in EVALUATOR_REGISTRY:
        return True, f"Unknown evaluator '{name}', skipping"

    entry = EVALUATOR_REGISTRY[name]

    if name == "length":
        return entry["fn"](copy_dict, constraints.get("length", {}))

    if name == "keywords":
        return entry["fn"](
            copy_dict,
            constraints.get("keywords_required", []),
            constraints.get("keywords_excluded", []),
        )

    if name == "punctuation":
        rules = constraints.get("punctuation_rules", {})
        if not rules:
            return True, "Punctuation check skipped: no rules configured"
        return entry["fn"](copy_dict, rules)

    if name == "off_brand":
        blocklist = constraints.get("off_brand_words", [])
        if not blocklist:
            return True, "Off-brand check skipped: no blocklist configured"
        return entry["fn"](copy_dict, blocklist)

    if name == "british_english":
        return entry["fn"](copy_dict)

    if name == "tone":
        tone_guidelines = constraints.get("tone_guidelines")
        if not tone_guidelines:
            return True, "Tone check skipped: no tone guidelines provided"
        examples = (eval_examples or {}).get("tone")
        return entry["fn"](client, model, copy_dict, tone_guidelines, examples=examples)

    if name == "coherence":
        examples = (eval_examples or {}).get("coherence")
        return entry["fn"](client, model, copy_dict, examples=examples)

    if name == "topic_relevance":
        topic = constraints.get("topic", "")
        persona = constraints.get("persona", "general audience")
        if not topic:
            return True, "Topic relevance check skipped: no topic specified"
        examples = (eval_examples or {}).get("topic_relevance")
        return entry["fn"](client, model, copy_dict, topic, persona, examples=examples)

    if name == "lexical_ordering":
        preferred_order = constraints.get("lexical_ordering", [])
        if not preferred_order:
            return True, "Lexical ordering check skipped: no ordering rules configured"
        examples = (eval_examples or {}).get("lexical_ordering")
        return entry["fn"](client, model, copy_dict, preferred_order, examples=examples)

    return True, f"Evaluator '{name}' has no dispatch logic"


def evaluate_copy(client, model, copy_dict, constraints, evaluator_sequence, eval_examples=None):
    for name in evaluator_sequence:
        passed, feedback = run_evaluator(name, client, model, copy_dict, constraints, eval_examples)
        if not passed:
            return False, name, feedback

    return True, None, "All evaluations passed"
