"""Divide — divides a by b."""

metadata = {
    "display_name": "Divide",
    "description": "Divides a by b. Returns an error if b is zero.",
    "category": "arithmetic",
    "color": "red",
}

inputs = [
    {"var_name": "a", "display_name": "Dividend (a)", "type": "number"},
    {"var_name": "b", "display_name": "Divisor (b)", "type": "number"},
]

outputs = [
    {"var_name": "result", "display_name": "Result", "type": "number"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    a = float(inputs.get("a", 0))
    b = float(inputs.get("b", 1))
    if b == 0:
        raise ValueError("Division by zero: 'b' must not be 0")
    return {"result": a / b}
