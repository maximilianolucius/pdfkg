"""
Generate summary report.
"""

from pathlib import Path

from pdfkg.config import Chunk, Mention


def generate_report(
    out_path: Path,
    sections: dict[str, dict],
    chunks: list[Chunk],
    mentions: list[Mention],
    figures: dict[str, str],
    tables: dict[str, str],
) -> None:
    """
    Generate a summary report in Markdown.

    Args:
        out_path: Output path for report.md.
        sections: Section tree dict.
        chunks: List of Chunk objects.
        mentions: List of Mention objects.
        figures: Figure index dict.
        tables: Table index dict.
    """
    lines = ["# PDF Knowledge Graph Report\n"]

    # Sections
    lines.append("## Sections")
    lines.append(f"- Total sections: {len(sections)}")
    if sections:
        max_level = max(s["level"] for s in sections.values())
        lines.append(f"- Maximum depth: {max_level}")
    lines.append("")

    # Chunks
    lines.append("## Chunks")
    lines.append(f"- Total chunks: {len(chunks)}")
    if chunks:
        avg_len = sum(len(c.text.split()) for c in chunks) / len(chunks)
        lines.append(f"- Average tokens (whitespace split): {avg_len:.1f}")
    lines.append("")

    # Figures
    lines.append("## Figures")
    lines.append(f"- Total figures indexed: {len(figures)}")
    if figures:
        lines.append("- Figure numbers: " + ", ".join(sorted(figures.keys())))
    lines.append("")

    # Tables
    lines.append("## Tables")
    lines.append(f"- Total tables indexed: {len(tables)}")
    if tables:
        lines.append("- Table numbers: " + ", ".join(sorted(tables.keys())))
    lines.append("")

    # Cross-references
    lines.append("## Cross-references")
    lines.append(f"- Total mentions: {len(mentions)}")
    resolved = sum(1 for m in mentions if m.target_id)
    unresolved = len(mentions) - resolved
    lines.append(f"- Resolved: {resolved}")
    lines.append(f"- Unresolved: {unresolved}")

    if unresolved > 0:
        lines.append("\n### Unresolved mentions")
        for m in mentions:
            if not m.target_id:
                lines.append(f"- `{m.raw_text}` (kind={m.kind}, hint={m.target_hint})")

    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
