"""Input Node — passes a static value into the workflow."""

metadata = {
    "display_name": "Input",
    "description": "Injects a static value into the workflow.",
    "category": "core",
    "color": "blue",
}

inputs = [
    {"var_name": "value", "display_name": "Value", "type": "any"},
]

outputs = [
    {"var_name": "value", "display_name": "Value", "type": "any"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    return {"value": inputs.get("value")}
