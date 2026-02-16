import json
from openai import OpenAI
from langflow.custom import Component
from langflow.io import StrInput, SecretStrInput, MultilineInput, Output
from langflow.schema.message import Message

from feedback_mapper import build_refiner_prompt


def refine_copy(client, model, copy_dict, evaluator_name, feedback, constraints):
    prompt = build_refiner_prompt(copy_dict, evaluator_name, feedback, constraints)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)


class CopyRefiner(Component):
    display_name = "Copy Refiner"
    description = "Refines marketing copy based on evaluation feedback using actionable instructions"
    icon = "refresh-cw"

    inputs = [
        MultilineInput(name="copy_json", display_name="Failed Copy JSON", required=True),
        StrInput(
            name="evaluator_name",
            display_name="Failed Evaluator Name",
            value="",
            info="Name of the evaluator that failed (e.g. length, tone, coherence). Used to generate targeted instructions.",
        ),
        MultilineInput(name="feedback", display_name="Evaluation Feedback", required=True),
        MultilineInput(
            name="constraints_json",
            display_name="Constraints JSON",
            required=True,
        ),
        SecretStrInput(name="api_key", display_name="OpenAI API Key", required=True),
        StrInput(name="model", display_name="Model", value="gpt-4o-mini"),
    ]

    outputs = [
        Output(display_name="Refined Copy", name="refined", method="refine"),
    ]

    def refine(self) -> Message:
        copy_dict = json.loads(self.copy_json)
        constraints = json.loads(self.constraints_json)
        client = OpenAI(api_key=self.api_key)

        refined = refine_copy(
            client, self.model, copy_dict,
            self.evaluator_name, self.feedback, constraints,
        )

        return Message(text=json.dumps(refined, indent=2))
