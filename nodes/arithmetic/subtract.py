"""Subtract — computes a minus b."""

metadata = {
    "display_name": "Subtract",
    "description": "Subtracts b from a.",
    "category": "arithmetic",
    "color": "orange",
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
    return {"result": a - b}
