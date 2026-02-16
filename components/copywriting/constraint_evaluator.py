import json
import os
from openai import OpenAI
from langflow.custom import Component
from langflow.io import StrInput, SecretStrInput, DropdownInput, MultilineInput, MessageTextInput, Output
from langflow.schema.message import Message

from evaluators import evaluate_copy, PRESETS


class ConstraintEvaluator(Component):
    display_name = "Constraint Evaluator"
    description = "Evaluates marketing copy against a configurable set of constraints including length, keywords, tone, coherence, topic relevance, lexical ordering, punctuation, and off-brand words"
    icon = "check-circle"

    inputs = [
        MultilineInput(name="copy_json", display_name="Copy JSON", required=True),
        MultilineInput(
            name="constraints_json",
            display_name="Constraints JSON",
            required=True,
            info=(
                '{"length": {"header": {"min": 30, "max": 60}}, '
                '"keywords_required": [], "keywords_excluded": [], '
                '"topic": "...", "persona": "...", '
                '"punctuation_rules": {}, "off_brand_words": [], '
                '"lexical_ordering": []}'
            ),
        ),
        MessageTextInput(
            name="tone_guidelines_json",
            display_name="Tone Guidelines",
            required=True,
            info="Accepts a file path (e.g. /app/langflow/tone.json), raw JSON, or a connection from Tone Extractor.",
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
            info="Only used when preset is 'custom'. Order matters -- first failure triggers feedback.",
        ),
        MultilineInput(
            name="eval_examples_json",
            display_name="Evaluator Few-Shot Examples JSON",
            value="{}",
            info='Optional few-shot examples keyed by evaluator name, e.g. {"tone": [{"copy": "...", "pass": true, "reasoning": "..."}]}',
        ),
        SecretStrInput(name="api_key", display_name="OpenAI API Key", required=True),
        StrInput(name="model", display_name="Model", value="gpt-4o-mini"),
    ]

    outputs = [
        Output(display_name="Evaluation Result", name="result", method="evaluate"),
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

    def evaluate(self) -> Message:
        copy_dict = json.loads(self.copy_json)
        constraints = json.loads(self.constraints_json)
        eval_examples = json.loads(self.eval_examples_json)

        tone_guidelines = self._parse_tone_guidelines()
        constraints["tone_guidelines"] = tone_guidelines

        if self.evaluator_preset == "custom":
            sequence = json.loads(self.custom_evaluators_json)
        else:
            sequence = PRESETS.get(self.evaluator_preset, PRESETS["standard"])

        client = OpenAI(api_key=self.api_key)

        passed, failed_evaluator, feedback = evaluate_copy(
            client, self.model, copy_dict, constraints, sequence, eval_examples
        )

        result = {
            "passed": passed,
            "failed_evaluator": failed_evaluator,
            "feedback": feedback,
        }
        return Message(text=json.dumps(result, indent=2))
