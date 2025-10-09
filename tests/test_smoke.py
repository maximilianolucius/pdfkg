"""
Smoke tests for pdfkg.
"""

import pytest
import networkx as nx

from pdfkg.config import Chunk, Mention
from pdfkg.graph import build_graph


def test_imports():
    """Test that all modules import successfully."""
    import pdfkg.config
    import pdfkg.parse_pdf
    import pdfkg.topology
    import pdfkg.chunking
    import pdfkg.embeds
    import pdfkg.xrefs
    import pdfkg.figtables
    import pdfkg.graph
    import pdfkg.report

    assert pdfkg.config is not None


def test_build_graph_synthetic():
    """Test graph construction with synthetic data."""
    # Synthetic data
    doc_id = "test_doc"
    pages = [{"page": 1, "text": "Page 1 text", "blocks": []}]
    sections = {
        "1": {"id": "1", "title": "Introduction", "level": 1, "page": 1, "children": []},
    }
    chunks = [
        Chunk(id="chunk1", section_id="1", page=1, text="This is a test chunk."),
    ]
    mentions = [
        Mention(
            source_chunk_id="chunk1",
            kind="section",
            raw_text="see section 1",
            target_hint="1",
            target_id="1",
        ),
    ]
    figures = {"1": "figure:1:p1"}
    tables = {}

    G = build_graph(doc_id, pages, sections, chunks, mentions, figures, tables)

    # Assertions
    assert G.number_of_nodes() > 0
    assert G.number_of_edges() > 0

    # Check node types
    node_types = {data["type"] for _, data in G.nodes(data=True)}
    assert "Document" in node_types
    assert "Page" in node_types
    assert "Section" in node_types
    assert "Paragraph" in node_types
    assert "Figure" in node_types

    # Check edge types
    edge_types = {data["type"] for _, _, data in G.edges(data=True)}
    assert "CONTAINS" in edge_types
    assert "LOCATED_ON" in edge_types
    assert "REFERS_TO" in edge_types


def test_chunk_creation():
    """Test chunk creation."""
    from pdfkg.chunking import section_chunks

    chunks = section_chunks(
        section_id="1.1",
        section_text="First sentence. Second sentence. Third sentence.",
        page_hint=5,
        max_tokens=10,
    )

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.section_id == "1.1"
        assert chunk.page == 5
        assert chunk.id is not None
