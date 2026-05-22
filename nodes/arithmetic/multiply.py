"""Multiply — multiplies a by b."""

metadata = {
    "display_name": "Multiply",
    "description": "Multiplies two numbers.",
    "category": "arithmetic",
    "color": "blue",
}

inputs = [
    {"var_name": "a", "display_name": "Number A", "type": "number"},
    {"var_name": "b", "display_name": "Number B", "type": "number"},
]

outputs = [
    {"var_name": "result", "display_name": "Result", "type": "number"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    a = float(inputs.get("a", 0))
    b = float(inputs.get("b", 0))
    return {"result": a * b}
