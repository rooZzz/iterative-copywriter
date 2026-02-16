import json
from openai import OpenAI
from langflow.custom import Component
from langflow.io import StrInput, SecretStrInput, MultilineInput, Output
from langflow.schema.message import Message


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


def check_tone(copy_dict, tone, api_key, model):
    client = OpenAI(api_key=api_key)
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
        "4. Decide if the copy meets the tone requirements",
        "",
        'Return a JSON object: {"pass": true/false, "reasoning": "...", "feedback": "..."}',
    ])

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    return result.get("pass", False), result.get("feedback", "Tone evaluation failed")


def evaluate_copy(copy_dict, constraints, api_key, model):
    length_constraints = constraints.get("length", {})
    required_kw = constraints.get("keywords_required", [])
    excluded_kw = constraints.get("keywords_excluded", [])
    tone = constraints.get("tone", "")

    passed, feedback = check_length(copy_dict, length_constraints)
    if not passed:
        return False, feedback

    passed, feedback = check_keywords(copy_dict, required_kw, excluded_kw)
    if not passed:
        return False, feedback

    if tone:
        passed, feedback = check_tone(copy_dict, tone, api_key, model)
        if not passed:
            return False, feedback

    return True, "All evaluations passed"


class ConstraintEvaluator(Component):
    display_name = "Constraint Evaluator"
    description = "Evaluates marketing copy against length, keyword, and tone-of-voice constraints"
    icon = "check-circle"

    inputs = [
        MultilineInput(name="copy_json", display_name="Copy JSON", required=True),
        MultilineInput(
            name="constraints_json",
            display_name="Constraints JSON",
            required=True,
            info='{"length": {"header": {"min": 30, "max": 60}}, "keywords_required": [], "keywords_excluded": [], "tone": "..."}',
        ),
        SecretStrInput(name="api_key", display_name="OpenAI API Key", required=True),
        StrInput(name="model", display_name="Model", value="gpt-4o-mini"),
    ]

    outputs = [
        Output(display_name="Evaluation Result", name="result", method="evaluate"),
    ]

    def evaluate(self) -> Message:
        copy_dict = json.loads(self.copy_json)
        constraints = json.loads(self.constraints_json)

        passed, feedback = evaluate_copy(
            copy_dict, constraints, self.api_key, self.model
        )

        result = {"passed": passed, "feedback": feedback}
        return Message(text=json.dumps(result, indent=2))
