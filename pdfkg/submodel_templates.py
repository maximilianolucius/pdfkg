"""Utilities for loading and working with submodel templates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "submodel_templates"


@dataclass
class SubmodelTemplate:
    """Container describing an AAS submodel template."""

    name: str
    display_name: str
    file_path: Path
    schema: Dict

    @property
    def json(self) -> str:
        """Return the template as pretty-printed JSON string."""
        return json.dumps(self.schema, indent=2)

    @property
    def default_text(self) -> str:
        """Pretty JSON string with trailing newline."""
        return self.json + '\n'


def _discover_templates() -> Dict[str, SubmodelTemplate]:
    templates: Dict[str, SubmodelTemplate] = {}
    if not TEMPLATE_DIR.exists():
        return templates

    for path in TEMPLATE_DIR.glob("*.json"):
        raw_name = path.stem.replace('_', ' ')
        key = ''.join(part.capitalize() for part in path.stem.split('_'))
        display_name = ' '.join(word.capitalize() for word in raw_name.split())
        with path.open('r', encoding='utf-8') as fh:
            schema = json.load(fh)
        templates[key] = SubmodelTemplate(name=raw_name, display_name=display_name, file_path=path, schema=schema)
    return templates


_TEMPLATES = _discover_templates()


def list_submodel_templates() -> List[str]:
    """Return sorted list of available submodel keys."""
    return sorted(_TEMPLATES.keys())


def get_template(key: str) -> SubmodelTemplate:
    """Get template by internal key."""
    try:
        return _TEMPLATES[key]
    except KeyError as exc:
        available = ", ".join(_TEMPLATES.keys()) or "<none>"
        raise KeyError(f"Unknown submodel template '{key}'. Available: {available}") from exc


def template_schema(key: str) -> Dict:
    """Return raw schema dictionary for a template key."""
    return get_template(key).schema


def template_json(key: str) -> str:
    """Return pretty-printed JSON string for a template key."""
    return get_template(key).json
