"""Read evaluation-only row identity declarations from annotation sidecars."""

from collections.abc import Mapping
from typing import Any


def top_level_identity_fields(block: Mapping[str, Any] | None) -> Mapping[str, tuple[str, ...]]:
    """Validate hints and select the top-level arrays scored by array-record F1.

    Nested array paths are retained by the test case for downstream scorers,
    but array-record F1 only scores top-level arrays. Identity bucketing is
    activated by the matcher only above its residual assignment cap.
    """
    if block is None:
        return {}
    if not isinstance(block, Mapping):
        raise ValueError("_identity_hints must be an object")
    result: dict[str, tuple[str, ...]] = {}
    for path, spec in block.items():
        if not isinstance(path, str) or any(len(part.strip()) == 0 for part in path.split(".")):
            raise ValueError("_identity_hints paths must be non-empty dotted array names")
        where = f"_identity_hints[{path!r}]"
        if isinstance(spec, Mapping):
            if set(spec) != {"fields"}:
                raise ValueError(f"{where} must contain only 'fields'")
            spec = spec["fields"]
        fields = [spec] if isinstance(spec, str) else spec
        if (
            not isinstance(fields, list)
            or len(fields) == 0
            or any(not isinstance(field, str) or len(field.strip()) == 0 for field in fields)
        ):
            raise ValueError(f"{where} must be a field name or a non-empty list of field names")
        if "." not in path:
            result[path] = tuple(fields)
    return result
