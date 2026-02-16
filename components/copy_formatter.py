import json
import re
from langflow.custom import Component
from langflow.io import StrInput, MultilineInput, Output
from langflow.schema.message import Message


def format_single_copy(copy_dict, structure):
    formatted = {}

    for key in ("header", "subheader"):
        if key not in copy_dict:
            continue

        text = copy_dict[key]
        text = text.replace(" and ", " & ")
        text = re.sub(r",(\s+)(and|or)\s", r"\1\2 ", text)

        if structure == "header" or key == "header":
            text = text.rstrip(".")

        text = text.strip()
        formatted[key] = text

    return formatted


class CopyFormatter(Component):
    display_name = "Copy Formatter"
    description = "Applies rule-based formatting to marketing copy (brand cleanup, punctuation, keyword substitution)"
    icon = "type"

    inputs = [
        MultilineInput(name="copy_json", display_name="Copy JSON", required=True),
        StrInput(
            name="copy_structure",
            display_name="Copy Structure",
            value="header",
            info="Either 'header' or 'header_subheader'",
        ),
    ]

    outputs = [
        Output(display_name="Formatted Copy", name="formatted", method="format_copy"),
    ]

    def format_copy(self) -> Message:
        copies = json.loads(self.copy_json)

        if isinstance(copies, dict):
            result = format_single_copy(copies, self.copy_structure)
        else:
            result = [format_single_copy(c, self.copy_structure) for c in copies]

        return Message(text=json.dumps(result, indent=2))
