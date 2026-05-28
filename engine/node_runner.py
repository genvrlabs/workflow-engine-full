"""Execute a node once or in batch after normalizing inputs."""

from __future__ import annotations

from typing import Any

from nodes.input_utils import (
    aggregate_outputs,
    detect_batches,
    normalize_inputs,
    normalize_value,
)


async def run_node(
    module: Any,
    uid: str,
    token: str,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """
    Run module.execute with deconstructed inputs.
    If a batched media input is detected, runs once per list item and
    aggregates outputs into lists.
    """
    input_specs = getattr(module, "inputs", [])
    batched, scalars, count = detect_batches(inputs, input_specs)

    if count == 0:
        run_inputs = normalize_inputs(inputs, input_specs)
        outputs = await module.execute(uid, token, run_inputs)
        return outputs if isinstance(outputs, dict) else {"value": outputs}

    results: list[dict[str, Any]] = []
    spec_by_name = {
        p["var_name"] if isinstance(p, dict) else p.var_name: (p if isinstance(p, dict) else p.to_dict())
        for p in input_specs
    }

    for i in range(count):
        run_inputs = dict(scalars)
        for name, items in batched.items():
            port = spec_by_name.get(name, {})
            run_inputs[name] = normalize_value(items[i], port)

        run_inputs = normalize_inputs(run_inputs, input_specs)
        out = await module.execute(uid, token, run_inputs)
        if not isinstance(out, dict):
            out = {"value": out}
        results.append(out)

    output_specs = getattr(module, "outputs", [])
    return aggregate_outputs(results, output_specs)
