import json
from openai import OpenAI
from langflow.custom import Component
from langflow.io import StrInput, IntInput, SecretStrInput, MultilineInput, Output
from langflow.schema.message import Message


class CopyGenerator(Component):
    display_name = "Copy Generator"
    description = "Generates a batch of marketing copies using OpenAI based on topic and constraints"
    icon = "pen-tool"

    inputs = [
        SecretStrInput(name="api_key", display_name="OpenAI API Key", required=True),
        StrInput(name="model", display_name="Model", value="gpt-4o-mini"),
        StrInput(name="topic", display_name="Topic", required=True),
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
    ]

    outputs = [
        Output(display_name="Generated Copies", name="copies", method="generate"),
    ]

    def _build_system_prompt(self, length_desc, required_kw, excluded_kw):
        if self.copy_structure == "header":
            structure_desc = 'Each copy must be a JSON object with a "header" key.'
        else:
            structure_desc = (
                'Each copy must be a JSON object with "header" and "subheader" keys. '
                "They should form a coherent message together."
            )

        parts = [
            "You are an expert marketing copywriter.",
            f"Generate exactly {self.batch_size} different marketing copies.",
            "",
            f"Topic: {self.topic}",
            f"Target audience: {self.persona}",
            f"Tone: {self.tone}",
            "",
            "Constraints:",
            f"- {structure_desc}",
            f"- Length: {length_desc}",
        ]

        if required_kw:
            parts.append(f"- Must include keywords: {', '.join(required_kw)}")
        if excluded_kw:
            parts.append(f"- Must NOT use these words: {', '.join(excluded_kw)}")

        parts.append("")
        parts.append(
            'Return a JSON object with a "copies" key containing an array of copy objects. '
            "Be creative and diverse in word choice and communication style."
        )

        return "\n".join(parts)

    def generate(self) -> Message:
        client = OpenAI(api_key=self.api_key)

        length_constraints = json.loads(self.length_constraints)
        required_kw = json.loads(self.keywords_required)
        excluded_kw = json.loads(self.keywords_excluded)

        length_desc = ", ".join(
            f"{k}: {v['min']}-{v['max']} characters"
            for k, v in length_constraints.items()
        )

        system_prompt = self._build_system_prompt(length_desc, required_kw, excluded_kw)

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
