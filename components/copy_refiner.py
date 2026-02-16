import json
from openai import OpenAI
from langflow.custom import Component
from langflow.io import StrInput, SecretStrInput, MultilineInput, Output
from langflow.schema.message import Message


def refine_copy(copy_dict, feedback, constraints, api_key, model):
    client = OpenAI(api_key=api_key)

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
    if constraints.get("keywords_excluded"):
        constraint_lines.append(
            f"Excluded words: {', '.join(constraints['keywords_excluded'])}"
        )

    prompt = "\n".join([
        "You are an expert marketing copywriter.",
        "The following copy failed evaluation and needs refinement.",
        "",
        f"Original copy:\n{json.dumps(copy_dict, indent=2)}",
        "",
        f"Evaluation feedback:\n{feedback}",
        "",
        "Constraints:",
        *[f"- {line}" for line in constraint_lines],
        "",
        "Generate a refined version that addresses the feedback while satisfying all constraints.",
        "Return ONLY a JSON object with the same keys as the original copy.",
    ])

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)


class CopyRefiner(Component):
    display_name = "Copy Refiner"
    description = "Refines marketing copy based on evaluation feedback using OpenAI"
    icon = "refresh-cw"

    inputs = [
        MultilineInput(name="copy_json", display_name="Failed Copy JSON", required=True),
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

        refined = refine_copy(
            copy_dict, self.feedback, constraints, self.api_key, self.model
        )

        return Message(text=json.dumps(refined, indent=2))
