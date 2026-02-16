import json
from openai import OpenAI
from langflow.custom import Component
from langflow.io import StrInput, SecretStrInput, MultilineInput, Output
from langflow.schema.message import Message


EXTRACTION_PROMPT = """You are a brand voice analyst. Analyze the following collection of marketing copy samples and extract structured tone of voice guidelines.

For each dimension below, identify the consistent patterns across all samples:

1. Voice attributes: 3-5 adjectives that describe the overall voice
2. Formality: the level of formality (e.g. "casual", "formal", "conversational but professional")
3. Sentence structure: what sentence patterns are preferred vs avoided
4. Vocabulary: preferred words, words to avoid, and the overall language register
5. Emotional register: the emotional quality of the writing
6. Punctuation preferences: any consistent punctuation patterns or avoidances
7. Examples: select 2-3 direct quotes from the samples that best represent the voice (on-brand), and construct 2-3 examples of what would clearly violate this voice (off-brand)

Return a JSON object with this exact structure:
{
  "voice_attributes": ["attr1", "attr2", "attr3"],
  "formality": "description of formality level",
  "sentence_structure": {
    "preferred": "description of preferred patterns",
    "avoid": "description of patterns to avoid"
  },
  "vocabulary": {
    "preferred_words": ["word1", "word2"],
    "avoid_words": ["word1", "word2"],
    "register": "description of language register"
  },
  "emotional_register": "description of emotional quality",
  "punctuation_preferences": "description of punctuation patterns",
  "examples": {
    "on_brand": ["example1", "example2"],
    "off_brand": ["example1", "example2"]
  }
}

Analyze ONLY the patterns present in the samples. Do not invent attributes that are not evidenced by the copy."""


class ToneExtractor(Component):
    display_name = "Tone Extractor"
    description = "Analyzes sample marketing copy to extract structured tone of voice guidelines."
    icon = "search"

    inputs = [
        SecretStrInput(name="api_key", display_name="OpenAI API Key", required=True),
        StrInput(
            name="model",
            display_name="Model",
            value="gpt-4o",
            info="Tone extraction is a one-time analysis that shapes all downstream output. Use the strongest model available.",
        ),
        MultilineInput(
            name="sample_copy",
            display_name="Sample Copy",
            required=True,
            info=(
                "Paste existing marketing copy to analyze. "
                "Accepts either a JSON array of copy objects "
                '(e.g. [{"header": "..."}, {"header": "...", "subheader": "..."}]) '
                "or plain text with one copy per line."
            ),
        ),
    ]

    outputs = [
        Output(display_name="Tone Guidelines", name="guidelines", method="extract"),
    ]

    def _parse_samples(self):
        text = self.sample_copy.strip()
        if text.startswith("["):
            samples = json.loads(text)
            lines = []
            for sample in samples:
                if isinstance(sample, dict):
                    lines.append(" | ".join(f"{k}: {v}" for k, v in sample.items()))
                else:
                    lines.append(str(sample))
            return "\n".join(lines)
        return text

    def extract(self) -> Message:
        client = OpenAI(api_key=self.api_key)

        formatted_samples = self._parse_samples()

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": f"Sample copy to analyze:\n\n{formatted_samples}"},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        guidelines = json.loads(response.choices[0].message.content)

        return Message(text=json.dumps(guidelines, indent=2))
