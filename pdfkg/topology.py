"""
Build document topology (section hierarchy) from ToC.
"""

import re
import unicodedata

HEADING_RX = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.+)$")


def slugify(text: str) -> str:
    """Convert text to a slug."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text.lower())
    text = re.sub(r"[-\s]+", "-", text)
    return text.strip("-")


def build_section_tree(toc: list[dict]) -> dict[str, dict]:
    """
    Build section tree from ToC.

    Args:
        toc: List of {level, title, page} from extract_toc.

    Returns:
        Dict mapping section_id -> {id, title, level, page, children: list[str]}.
    """
    sections = {}
    parent_stack = []

    for entry in toc:
        level = entry["level"]
        title = entry["title"]
        page = entry["page"]

        # Parse section number from title
        match = HEADING_RX.match(title)
        if match:
            section_id = match.group(1)
            clean_title = match.group(2).strip()
        else:
            # Fallback to slugified title
            section_id = slugify(title)
            clean_title = title

        # Build section node
        section = {
            "id": section_id,
            "title": clean_title,
            "level": level,
            "page": page,
            "children": [],
        }
        sections[section_id] = section

        # Maintain parent-child relationships
        while parent_stack and parent_stack[-1]["level"] >= level:
            parent_stack.pop()

        if parent_stack:
            parent_stack[-1]["children"].append(section_id)

        parent_stack.append(section)

    return sections


def infer_headings_from_text(pages: list[dict]) -> dict[str, dict]:
    """
    Infer section headings from page text when ToC is missing.

    TODO: Implement heuristic heading detection (large font, bold, numbering).
    For now, returns empty dict.
    """
    # Placeholder for future implementation
    return {}
