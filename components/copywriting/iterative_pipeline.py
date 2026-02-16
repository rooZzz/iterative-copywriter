import json
import os
from openai import OpenAI
from langflow.custom import Component
from langflow.io import (
    StrInput,
    IntInput,
    BoolInput,
    FloatInput,
    SecretStrInput,
    DropdownInput,
    MultilineInput,
    MessageTextInput,
    Output,
)
from langflow.schema.message import Message

from evaluators import evaluate_copy, PRESETS
from feedback_mapper import build_refiner_prompt
from copy_formatter import format_single_copy, DEFAULT_RULES
from deduplication import deduplicate_copies
from human_review import load_feedback_log, get_negative_examples
from tone_guidelines import format_guidelines_for_prompt


def _generate_copies(client, model, topic, persona, tone_guidelines, copy_structure,
                     length_constraints, keywords_required, keywords_excluded,
                     batch_size, examples=None, negative_examples=None):
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

    guidelines_block = format_guidelines_for_prompt(tone_guidelines)

    parts = [
        "You are an expert marketing copywriter writing for a British audience.",
        "All output MUST use British English spelling and conventions (e.g. 'personalised' not 'personalized', 'colour' not 'color', 'organisation' not 'organization').",
        f"Generate exactly {batch_size} different marketing copies.",
        "",
        f"Topic: {topic}",
        f"Target audience: {persona}",
        "",
        "Tone of voice guidelines:",
        guidelines_block,
        "",
        "Constraints:",
        f"- {structure_desc}",
        f"- Length: {length_desc}",
    ]

    if keywords_required:
        parts.append(f"- Must include keywords: {', '.join(keywords_required)}")
    if keywords_excluded:
        parts.append(f"- Must NOT use these words: {', '.join(keywords_excluded)}")

    on_brand = tone_guidelines.get("examples", {}).get("on_brand", [])
    off_brand = tone_guidelines.get("examples", {}).get("off_brand", [])

    if on_brand or examples:
        parts.append("")
        parts.append("IMPORTANT: Your copy MUST sound like these on-brand examples:")
        for i, ex in enumerate(on_brand, 1):
            parts.append(f"  On-brand {i}: {ex}")
        for i, ex in enumerate(examples or [], len(on_brand) + 1):
            parts.append(f"  On-brand {i}: {json.dumps(ex)}")

    if off_brand:
        parts.append("")
        parts.append("DO NOT write copy that sounds like these off-brand examples:")
        for i, ex in enumerate(off_brand, 1):
            parts.append(f"  Off-brand {i}: {ex}")

    if negative_examples:
        parts.append("")
        parts.append("Avoid producing copies like these (rejected in past reviews):")
        for i, neg in enumerate(negative_examples, 1):
            reason = neg.get("reason", "")
            copy_str = json.dumps(neg.get("copy", {}))
            parts.append(f"Bad example {i}: {copy_str} (reason: {reason})")

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


def _refine_copy(client, model, copy_dict, evaluator_name, feedback, constraints):
    prompt = build_refiner_prompt(copy_dict, evaluator_name, feedback, constraints)

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
        "Implements the iterative refinement framework with configurable evaluators, "
        "few-shot examples, feedback mapping, and deduplication."
    )
    icon = "repeat"

    inputs = [
        SecretStrInput(name="api_key", display_name="OpenAI API Key", required=True),
        StrInput(
            name="generation_model",
            display_name="Generation Model",
            value="gpt-4o-mini",
            info="Model for generating and refining copy. Mini is fine here — creativity benefits from high temperature, not model size.",
        ),
        StrInput(
            name="evaluation_model",
            display_name="Evaluation Model",
            value="gpt-4o",
            info="Model for tone, coherence, and topic relevance checks. Larger models are much stricter and catch more issues.",
        ),
        MessageTextInput(
            name="topic",
            display_name="Topic",
            required=True,
            info="The service or product to write copy about. Accepts a Chat Input connection or typed text.",
        ),
        StrInput(name="persona", display_name="Target Persona", value="general audience"),
        MessageTextInput(
            name="tone_guidelines_json",
            display_name="Tone Guidelines",
            required=True,
            info="Accepts a file path (e.g. /app/langflow/tone.json), raw JSON, or a connection from Tone Extractor.",
        ),
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
        MultilineInput(
            name="examples_json",
            display_name="Few-Shot Examples JSON",
            value="[]",
            info='Optional JSON array of example copies for in-context learning, e.g. [{"header": "Get Free Delivery Today"}]',
        ),
        IntInput(name="batch_size", display_name="Batch Size", value=5),
        IntInput(
            name="target_accepted",
            display_name="Target Accepted",
            value=0,
            info="Stop early once this many copies are accepted. 0 = process all.",
        ),
        IntInput(
            name="max_refinements",
            display_name="Max Refinement Attempts",
            value=2,
            info="Maximum number of refine attempts per copy (paper recommends 1-2)",
        ),
        DropdownInput(
            name="evaluator_preset",
            display_name="Evaluator Preset",
            options=["basic", "standard", "full", "custom"],
            value="standard",
            info="basic: length+keywords | standard: +tone | full: all evaluators | custom: use JSON below",
        ),
        MultilineInput(
            name="custom_evaluators_json",
            display_name="Custom Evaluator Sequence JSON",
            value='["length", "keywords", "tone"]',
            info='Only used when preset is "custom". Order matters -- first failure triggers refinement.',
        ),
        MultilineInput(
            name="eval_examples_json",
            display_name="Evaluator Few-Shot Examples JSON",
            value="{}",
            info='Optional few-shot examples keyed by evaluator, e.g. {"tone": [{"copy": "...", "pass": true, "reasoning": "..."}]}',
        ),
        MultilineInput(
            name="off_brand_words_json",
            display_name="Off-Brand Words JSON",
            value="[]",
            info="Words to reject beyond excluded keywords (slang, competitor names, etc.)",
        ),
        MultilineInput(
            name="punctuation_rules_json",
            display_name="Punctuation Rules JSON",
            value="{}",
            info='e.g. {"no_exclamation_marks": true, "no_ellipsis": true, "no_all_caps_words": true, "max_commas_per_component": 2}',
        ),
        MultilineInput(
            name="lexical_ordering_json",
            display_name="Lexical Ordering Rules JSON",
            value="[]",
            info='e.g. [{"before": "brand name", "after": "product category"}]',
        ),
        MultilineInput(
            name="formatting_rules_json",
            display_name="Formatting Rules JSON",
            value=json.dumps(DEFAULT_RULES, indent=2),
            info="Toggle formatting rules on/off",
        ),
        BoolInput(
            name="enable_dedup",
            display_name="Enable Deduplication",
            value=True,
            info="Remove near-duplicate copies from accepted results",
        ),
        FloatInput(
            name="similarity_threshold",
            display_name="Similarity Threshold",
            value=0.85,
            info="Copies above this similarity score are considered duplicates (0.0-1.0)",
        ),
        StrInput(
            name="feedback_log_path",
            display_name="Feedback Log Path",
            value="",
            info="Path to feedback_log.json from past human reviews. Leave blank to skip.",
        ),
    ]

    outputs = [
        Output(display_name="Pipeline Results", name="results", method="run_pipeline"),
        Output(display_name="Chat Output", name="chat_output", method="run_pipeline_chat"),
    ]

    def _parse_tone_guidelines(self):
        raw = self.tone_guidelines_json
        if hasattr(raw, "text"):
            raw = raw.text
        raw = str(raw).strip() if raw else ""
        if not raw:
            raise ValueError(
                "Tone Guidelines input is empty. Provide a file path, paste JSON, "
                "or connect the Tone Extractor output."
            )
        if raw.startswith("/") and os.path.isfile(raw):
            with open(raw) as f:
                data = json.load(f)
        else:
            data = json.loads(raw)
        if isinstance(data, dict) and "message" in data and len(data) == 1:
            data = json.loads(data["message"])
        return data

    def _resolve_topic(self):
        raw = self.topic
        if hasattr(raw, "text"):
            raw = raw.text
        return str(raw).strip()

    def _execute_pipeline(self):
        client = OpenAI(api_key=self.api_key)
        topic = self._resolve_topic()

        tone_guidelines = self._parse_tone_guidelines()

        length_constraints = json.loads(self.length_constraints)
        required_kw = json.loads(self.keywords_required)
        excluded_kw = json.loads(self.keywords_excluded)
        examples = json.loads(self.examples_json)
        eval_examples = json.loads(self.eval_examples_json)
        off_brand_words = json.loads(self.off_brand_words_json)
        punctuation_rules = json.loads(self.punctuation_rules_json)
        lexical_ordering = json.loads(self.lexical_ordering_json)
        formatting_rules = json.loads(self.formatting_rules_json)

        if self.evaluator_preset == "custom":
            evaluator_sequence = json.loads(self.custom_evaluators_json)
        else:
            evaluator_sequence = list(PRESETS.get(self.evaluator_preset, PRESETS["standard"]))

        constraints = {
            "topic": topic,
            "persona": self.persona,
            "tone_guidelines": tone_guidelines,
            "length": length_constraints,
            "keywords_required": required_kw,
            "keywords_excluded": excluded_kw,
            "off_brand_words": off_brand_words,
            "punctuation_rules": punctuation_rules,
            "lexical_ordering": lexical_ordering,
        }

        negative_examples = []
        if self.feedback_log_path:
            feedback_log = load_feedback_log(self.feedback_log_path)
            negative_examples = get_negative_examples(feedback_log)

        raw_copies = _generate_copies(
            client, self.generation_model, topic, self.persona, tone_guidelines,
            self.copy_structure, length_constraints, required_kw, excluded_kw,
            self.batch_size, examples=examples, negative_examples=negative_examples,
        )

        accepted = []
        rejected = []
        log = []

        for i, raw_copy in enumerate(raw_copies):
            if self.target_accepted > 0 and len(accepted) >= self.target_accepted:
                break

            formatted = format_single_copy(raw_copy, self.copy_structure, formatting_rules)
            copy_log = {"index": i, "original": raw_copy, "attempts": []}
            passed = False

            for attempt in range(self.max_refinements + 1):
                success, failed_evaluator, feedback = evaluate_copy(
                    client, self.evaluation_model, formatted, constraints,
                    evaluator_sequence, eval_examples,
                )

                copy_log["attempts"].append({
                    "attempt": attempt,
                    "copy": dict(formatted),
                    "passed": success,
                    "failed_evaluator": failed_evaluator,
                    "feedback": feedback,
                })

                if success:
                    accepted.append(formatted)
                    passed = True
                    break

                if attempt < self.max_refinements:
                    refined = _refine_copy(
                        client, self.generation_model, formatted,
                        failed_evaluator or "", feedback, constraints,
                    )
                    formatted = format_single_copy(refined, self.copy_structure, formatting_rules)

            if not passed:
                rejected.append(formatted)

            log.append(copy_log)

        if self.enable_dedup and len(accepted) > 1:
            before_dedup = len(accepted)
            accepted = deduplicate_copies(
                accepted,
                strategy="ngram",
                threshold=self.similarity_threshold,
            )
            dedup_removed = before_dedup - len(accepted)
        else:
            dedup_removed = 0

        total = len(raw_copies)
        success_rate = (len(accepted) / total * 100) if total > 0 else 0.0

        return {
            "topic": topic,
            "summary": {
                "total_generated": total,
                "accepted": len(accepted),
                "rejected": len(rejected),
                "duplicates_removed": dedup_removed,
                "success_rate": round(success_rate, 2),
                "evaluator_preset": self.evaluator_preset,
                "evaluator_sequence": evaluator_sequence,
            },
            "accepted_copies": accepted,
            "rejected_copies": rejected,
            "detailed_log": log,
        }

    def run_pipeline(self) -> Message:
        result = self._execute_pipeline()
        return Message(text=json.dumps(result, indent=2))

    def run_pipeline_chat(self) -> Message:
        result = self._execute_pipeline()
        accepted = result["accepted_copies"]
        rejected_count = result["summary"]["rejected"]
        total = result["summary"]["total_generated"]
        topic = result["topic"]

        lines = []
        lines.append(f"**Copy for: {topic}**")
        lines.append("")
        lines.append(f"Generated {total} variants — {len(accepted)} accepted, {rejected_count} rejected.")
        lines.append("")

        if not accepted:
            lines.append("No copies passed evaluation. Try adjusting the topic or relaxing the evaluator preset.")
        else:
            for i, copy in enumerate(accepted, 1):
                lines.append(f"---")
                lines.append(f"**Option {i}**")
                lines.append("")
                for key, value in copy.items():
                    label = key.replace("_", " ").title()
                    lines.append(f"**{label}:** {value}")
                lines.append("")

        return Message(text="\n".join(lines))
