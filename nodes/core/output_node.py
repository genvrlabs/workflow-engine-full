"""Output Node — collects a final value and labels it."""

metadata = {
    "display_name": "Output",
    "description": "Collects a value as a named workflow output.",
    "category": "core",
    "color": "purple",
}

inputs = [
    {"var_name": "value", "display_name": "Value", "type": "any"},
    {"var_name": "label", "display_name": "Label", "type": "text"},
]

outputs = [
    {"var_name": "label", "display_name": "Label", "type": "text"},
    {"var_name": "value", "display_name": "Value", "type": "any"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    return {
        "label": inputs.get("label", "output"),
        "value": inputs.get("value"),
    }
