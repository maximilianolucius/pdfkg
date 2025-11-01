"""Helpers for enforcing submodel template structure and metadata."""

from __future__ import annotations

from typing import Any, Dict, Set

from pdfkg.submodel_templates import template_schema


def sanitize_to_schema(schema: Any, value: Any) -> Any:
    """Coerce ``value`` to match the structure of ``schema`` while removing extra keys."""
    if isinstance(schema, dict):
        source = value if isinstance(value, dict) else {}
        cleaned: Dict[str, Any] = {}
        for key, subschema in schema.items():
            cleaned[key] = sanitize_to_schema(subschema, source.get(key))
        return cleaned

    if isinstance(schema, list):
        item_schema = schema[0] if schema else None
        if not isinstance(value, list) or item_schema is None:
            return []
        return [sanitize_to_schema(item_schema, item) for item in value if item is not None]

    # Scalar leaf value
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        # Convert complex leftovers to string representation to avoid nested structures
        return str(value)
    return value


def _collect_metadata_paths(schema: Any, prefix: str = "") -> Set[str]:
    paths: Set[str] = set()
    if isinstance(schema, dict):
        for key, subschema in schema.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            paths.update(_collect_metadata_paths(subschema, new_prefix))
    elif isinstance(schema, list):
        list_prefix = f"{prefix}[]" if prefix else "[]"
        paths.add(list_prefix)
        if schema:
            paths.update(_collect_metadata_paths(schema[0], prefix))
    else:
        if prefix:
            paths.add(prefix)
    return paths


def sanitize_submodel_data(template_key: str, data: Any) -> Any:
    """Sanitize arbitrary data to match the JSON structure of the given template."""
    schema = template_schema(template_key)
    return sanitize_to_schema(schema, data)


def filter_metadata(template_key: str, metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Filter metadata entries so only template-defined paths remain."""
    allowed = _collect_metadata_paths(template_schema(template_key))
    if not metadata:
        return {}
    return {path: info for path, info in metadata.items() if path in allowed}
