"""
Base node contract for the GenVR Workflow Engine.

Every node module must export:
  - inputs:  list of {"var_name": str, "display_name": str, "type": str}
  - outputs: list of {"var_name": str, "display_name": str, "type": str}
  - metadata: dict with display_name, description, category, color
  - execute(uid, token, inputs) -> dict

The execute function receives:
  - uid:    user id (str)
  - token:  auth token (str)
  - inputs: dict mapping var_name -> value (already resolved from the graph)

It must return a dict mapping output var_name -> value.
"""

from abc import ABC, abstractmethod
from typing import Any


class NodePort:
    def __init__(self, var_name: str, display_name: str, type: str = "any"):
        self.var_name = var_name
        self.display_name = display_name
        self.type = type

    def to_dict(self) -> dict:
        return {
            "var_name": self.var_name,
            "display_name": self.display_name,
            "type": self.type,
        }


class BaseNode(ABC):
    """Optional base class — nodes can also just be plain modules with the required exports."""

    inputs: list[NodePort] = []
    outputs: list[NodePort] = []
    metadata: dict = {
        "display_name": "Unnamed Node",
        "description": "",
        "category": "general",
        "color": "gray",
    }

    @abstractmethod
    async def execute(self, uid: str, token: str, inputs: dict[str, Any]) -> dict[str, Any]:
        ...
