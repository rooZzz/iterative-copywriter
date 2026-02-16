import json
import re
from langflow.custom import Component
from langflow.io import StrInput, MultilineInput, Output
from langflow.schema.message import Message


DEFAULT_RULES = {
    "replace_and_with_ampersand": True,
    "remove_serial_commas": True,
    "strip_trailing_periods": True,
}


def format_single_copy(copy_dict, structure, rules=None):
    active_rules = dict(DEFAULT_RULES)
    if rules:
        active_rules.update(rules)

    formatted = {}

    for key in ("header", "subheader"):
        if key not in copy_dict:
            continue

        text = copy_dict[key]

        if active_rules.get("replace_and_with_ampersand"):
            text = text.replace(" and ", " & ")

        if active_rules.get("remove_serial_commas"):
            text = re.sub(r",(\s+)(and|or)\s", r"\1\2 ", text)

        if active_rules.get("strip_trailing_periods"):
            if structure == "header" or key == "header":
                text = text.rstrip(".")

        text = text.strip()
        formatted[key] = text

    return formatted


class CopyFormatter(Component):
    display_name = "Copy Formatter"
    description = "Applies configurable rule-based formatting to marketing copy (brand cleanup, punctuation, keyword substitution)"
    icon = "type"

    inputs = [
        MultilineInput(name="copy_json", display_name="Copy JSON", required=True),
        StrInput(
            name="copy_structure",
            display_name="Copy Structure",
            value="header",
            info="Either 'header' or 'header_subheader'",
        ),
        MultilineInput(
            name="formatting_rules_json",
            display_name="Formatting Rules JSON",
            value=json.dumps(DEFAULT_RULES, indent=2),
            info="Toggle formatting rules on/off. All default to true.",
        ),
    ]

    outputs = [
        Output(display_name="Formatted Copy", name="formatted", method="format_copy"),
    ]

    def format_copy(self) -> Message:
        copies = json.loads(self.copy_json)
        rules = json.loads(self.formatting_rules_json)

        if isinstance(copies, dict):
            result = format_single_copy(copies, self.copy_structure, rules)
        else:
            result = [format_single_copy(c, self.copy_structure, rules) for c in copies]

        return Message(text=json.dumps(result, indent=2))
