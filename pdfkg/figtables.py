"""
Figure and table caption indexing.
"""

import re

FIG_RX = re.compile(r"^\s*(?:Fig(?:\.|ure)?)\s*[: ]*([0-9]+[A-Z]?)\b", re.I | re.M)
TAB_RX = re.compile(r"^\s*(?:Tab(?:\.|le)?)\s*[: ]*([0-9]+[A-Z]?)\b", re.I | re.M)


def index_figures_tables(pages: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
    """
    Index figures and tables from page text.

    Args:
        pages: List of page dicts from extract_pages.

    Returns:
        Tuple of (figures, tables) where each is dict: number -> stable id.
    """
    figures = {}
    tables = {}

    for page_data in pages:
        page_num = page_data["page"]
        text = page_data["text"]

        # Find figures
        for match in FIG_RX.finditer(text):
            fig_num = match.group(1).upper()
            fig_id = f"figure:{fig_num}:p{page_num}"
            if fig_num not in figures:
                figures[fig_num] = fig_id

        # Find tables
        for match in TAB_RX.finditer(text):
            tab_num = match.group(1).upper()
            tab_id = f"table:{tab_num}:p{page_num}"
            if tab_num not in tables:
                tables[tab_num] = tab_id

    return figures, tables
