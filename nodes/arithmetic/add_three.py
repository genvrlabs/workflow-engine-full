"""Add Three Numbers — adds inputs a, b, and c."""

metadata = {
    "display_name": "Add Three Numbers",
    "description": "Adds three numbers and returns the result.",
    "category": "arithmetic",
    "color": "green",
}

inputs = [
    {"var_name": "a", "display_name": "Number A", "type": "number"},
    {"var_name": "b", "display_name": "Number B", "type": "number"},
    {"var_name": "c", "display_name": "Number C", "type": "number"},
]

outputs = [
    {"var_name": "result", "display_name": "Result", "type": "number"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    a = float(inputs.get("a", 0))
    b = float(inputs.get("b", 0))
    c = float(inputs.get("c", 0))
    return {"result": a + b + c}
