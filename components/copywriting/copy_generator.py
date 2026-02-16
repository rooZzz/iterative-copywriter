import json
import os
from openai import OpenAI
from langflow.custom import Component
from langflow.io import StrInput, IntInput, SecretStrInput, MultilineInput, MessageTextInput, Output
from langflow.schema.message import Message

from tone_guidelines import format_guidelines_for_prompt


class CopyGenerator(Component):
    display_name = "Copy Generator"
    description = "Generates a batch of marketing copies using OpenAI based on topic and constraints"
    icon = "pen-tool"

    inputs = [
        SecretStrInput(name="api_key", display_name="OpenAI API Key", required=True),
        StrInput(name="model", display_name="Model", value="gpt-4o-mini"),
        StrInput(name="topic", display_name="Topic", required=True),
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
    ]

    outputs = [
        Output(display_name="Generated Copies", name="copies", method="generate"),
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

    def _build_system_prompt(self, tone_guidelines, length_desc, required_kw, excluded_kw, examples):
        if self.copy_structure == "header":
            structure_desc = 'Each copy must be a JSON object with a "header" key.'
        else:
            structure_desc = (
                'Each copy must be a JSON object with "header" and "subheader" keys. '
                "They should form a coherent message together."
            )

        guidelines_block = format_guidelines_for_prompt(tone_guidelines)

        parts = [
            "You are an expert marketing copywriter writing for a British audience.",
            "All output MUST use British English spelling and conventions (e.g. 'personalised' not 'personalized', 'colour' not 'color', 'organisation' not 'organization').",
            f"Generate exactly {self.batch_size} different marketing copies.",
            "",
            f"Topic: {self.topic}",
            f"Target audience: {self.persona}",
            "",
            "Tone of voice guidelines:",
            guidelines_block,
            "",
            "Constraints:",
            f"- {structure_desc}",
            f"- Length: {length_desc}",
        ]

        if required_kw:
            parts.append(f"- Must include keywords: {', '.join(required_kw)}")
        if excluded_kw:
            parts.append(f"- Must NOT use these words: {', '.join(excluded_kw)}")

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

        parts.append("")
        parts.append(
            'Return a JSON object with a "copies" key containing an array of copy objects. '
            "Be creative and diverse in word choice and communication style."
        )

        return "\n".join(parts)

    def generate(self) -> Message:
        client = OpenAI(api_key=self.api_key)

        tone_guidelines = self._parse_tone_guidelines()

        length_constraints = json.loads(self.length_constraints)
        required_kw = json.loads(self.keywords_required)
        excluded_kw = json.loads(self.keywords_excluded)
        examples = json.loads(self.examples_json)

        length_desc = ", ".join(
            f"{k}: {v['min']}-{v['max']} characters"
            for k, v in length_constraints.items()
        )

        system_prompt = self._build_system_prompt(
            tone_guidelines, length_desc, required_kw, excluded_kw, examples,
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate {self.batch_size} copies about: {self.topic}"},
            ],
            temperature=0.9,
            response_format={"type": "json_object"},
        )

        raw = json.loads(response.choices[0].message.content)
        copies = raw.get("copies", [raw] if isinstance(raw, dict) else raw)

        return Message(text=json.dumps(copies, indent=2))
