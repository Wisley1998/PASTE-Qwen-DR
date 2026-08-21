"""Canonical, immutable tool invocation values.

The scheduler matches on the complete canonical JSON argument object.  Mapping
key order and JSON whitespace are irrelevant; values, list order, and extra
arguments are not.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from typing import Any


def _json_value(value: Any, path: str = "arguments") -> Any:
    """Return a JSON-compatible copy while rejecting lossy conversions."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        copied = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key: {key!r}")
            copied[key] = _json_value(child, f"{path}.{key}")
        return copied
    if isinstance(value, (list, tuple)):
        return [_json_value(child, f"{path}[{index}]") for index, child in enumerate(value)]
    raise TypeError(f"{path} contains a non-JSON value: {type(value).__name__}")


def canonicalize_arguments(arguments: Mapping[str, Any]) -> str:
    """Serialize arguments into the exact comparison form used by PASTE."""

    if not isinstance(arguments, Mapping):
        raise TypeError("tool arguments must be a mapping")
    copied = _json_value(arguments)
    try:
        return json.dumps(
            copied,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"tool arguments are not canonical JSON: {exc}") from exc


@dataclass(frozen=True, init=False)
class Invocation:
    """An invocation whose arguments cannot change after it is scheduled."""

    tool_name: str
    _canonical_arguments: str

    def __init__(self, tool_name: str, arguments: Mapping[str, Any]) -> None:
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError("tool_name must be a non-empty string")
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "_canonical_arguments", canonicalize_arguments(arguments))

    @property
    def arguments(self) -> dict[str, Any]:
        """Return a fresh copy of the canonical argument object."""

        value = json.loads(self._canonical_arguments)
        assert isinstance(value, dict)
        return value

    @property
    def canonical_arguments(self) -> str:
        return self._canonical_arguments

    @property
    def key(self) -> tuple[str, str]:
        return self.tool_name, self._canonical_arguments

    def to_dict(self) -> dict[str, Any]:
        return {"tool_name": self.tool_name, "arguments": self.arguments}

