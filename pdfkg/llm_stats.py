"""Simple in-memory counters for LLM call usage."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, Tuple

_CALL_COUNTS: Counter[Tuple[str, str, str]] = Counter()


def record_call(provider: str, phase: str, label: str) -> None:
    """Record a single LLM call."""
    key = (provider, phase, label)
    _CALL_COUNTS[key] += 1


def summary() -> Dict[str, Dict[str, Dict[str, int]]]:
    """Return nested dict provider -> phase -> label -> count."""
    report: Dict[str, Dict[str, Dict[str, int]]] = {}
    for (provider, phase, label), count in _CALL_COUNTS.items():
        report.setdefault(provider, {}).setdefault(phase, {})[label] = count
    return report


def summary_lines() -> Iterable[str]:
    """Return a list of formatted strings summarizing counters."""
    if not _CALL_COUNTS:
        return ["No LLM calls recorded yet."]

    lines = ["LLM Call Summary:"]
    for provider, phase_map in summary().items():
        lines.append(f"- Provider: {provider}")
        for phase, label_map in phase_map.items():
            lines.append(f"  • Phase: {phase}")
            for label, count in sorted(label_map.items()):
                lines.append(f"    - {label}: {count}")
    return lines


def reset() -> None:
    """Reset counters to zero."""
    _CALL_COUNTS.clear()
