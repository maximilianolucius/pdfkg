"""
Knowledge graph construction using NetworkX.
"""

from pathlib import Path
import json

import networkx as nx
import orjson

from pdfkg.config import Chunk, Mention


def build_graph(
    doc_id: str,
    pages: list[dict],
    sections: dict[str, dict],
    chunks: list[Chunk],
    mentions: list[Mention],
    figures: dict[str, str],
    tables: dict[str, str],
) -> nx.MultiDiGraph:
    """
    Build a MultiDiGraph with nodes for pages, sections, paragraphs, figures, tables.

    Node types: Page, Section, Paragraph, Figure, Table
    Edge types: CONTAINS, LOCATED_ON, REFERS_TO

    Args:
        doc_id: Document identifier.
        pages: List of page dicts.
        sections: Section tree dict.
        chunks: List of Chunk objects.
        mentions: List of resolved Mention objects.
        figures: Figure index dict.
        tables: Table index dict.

    Returns:
        NetworkX MultiDiGraph.
    """
    G = nx.MultiDiGraph()

    # Add document node
    G.add_node(doc_id, type="Document", label=doc_id)

    # Add page nodes
    for page_data in pages:
        page_num = page_data["page"]
        page_id = f"page:{page_num}"
        G.add_node(page_id, type="Page", label=f"Page {page_num}", page=page_num)
        G.add_edge(doc_id, page_id, type="CONTAINS", kind="page")

    # Add section nodes
    for sec_id, sec in sections.items():
        G.add_node(
            sec_id,
            type="Section",
            label=sec["title"],
            level=sec["level"],
            page=sec["page"],
        )
        # Section located on page
        page_id = f"page:{sec['page']}"
        if G.has_node(page_id):
            G.add_edge(page_id, sec_id, type="LOCATED_ON", kind="section")

        # Parent-child relationships
        for child_id in sec.get("children", []):
            G.add_edge(sec_id, child_id, type="CONTAINS", kind="subsection")

    # Add paragraph nodes (chunks)
    for chunk in chunks:
        para_id = chunk.id
        G.add_node(
            para_id,
            type="Paragraph",
            label=chunk.text[:50] + "..." if len(chunk.text) > 50 else chunk.text,
            text=chunk.text,
            section_id=chunk.section_id,
            page=chunk.page,
        )
        # Paragraph located on page
        page_id = f"page:{chunk.page}"
        if G.has_node(page_id):
            G.add_edge(page_id, para_id, type="LOCATED_ON", kind="paragraph")
        # Paragraph contained in section
        if chunk.section_id in sections:
            G.add_edge(chunk.section_id, para_id, type="CONTAINS", kind="paragraph")

    # Add figure nodes
    for fig_num, fig_id in figures.items():
        G.add_node(fig_id, type="Figure", label=f"Figure {fig_num}", number=fig_num)
        # Extract page from fig_id (format: figure:{num}:p{page})
        parts = fig_id.split(":p")
        if len(parts) == 2:
            page_num = int(parts[1])
            page_id = f"page:{page_num}"
            if G.has_node(page_id):
                G.add_edge(page_id, fig_id, type="LOCATED_ON", kind="figure")

    # Add table nodes
    for tab_num, tab_id in tables.items():
        G.add_node(tab_id, type="Table", label=f"Table {tab_num}", number=tab_num)
        parts = tab_id.split(":p")
        if len(parts) == 2:
            page_num = int(parts[1])
            page_id = f"page:{page_num}"
            if G.has_node(page_id):
                G.add_edge(page_id, tab_id, type="LOCATED_ON", kind="table")

    # Add cross-reference edges
    for mention in mentions:
        if mention.target_id and G.has_node(mention.target_id):
            G.add_edge(
                mention.source_chunk_id,
                mention.target_id,
                type="REFERS_TO",
                kind=mention.kind,
                raw=mention.raw_text,
            )

    return G


def to_cypher(G: nx.MultiDiGraph, path: Path) -> None:
    """
    Export graph as Cypher MERGE statements.

    All nodes labeled as `Node` with `type` property.
    Edges carry `type`, `kind`, and `raw` properties.

    Args:
        G: NetworkX graph.
        path: Output path for .cypher file.
    """
    lines = []

    # Nodes
    for node_id, attrs in G.nodes(data=True):
        props = {"id": node_id}
        props.update(attrs)
        # Escape values
        prop_str = ", ".join(
            [f"{k}: {json.dumps(v)}" for k, v in props.items() if v is not None]
        )
        lines.append(f"MERGE (n:Node {{{prop_str}}});")

    # Edges
    for u, v, key, attrs in G.edges(data=True, keys=True):
        edge_type = attrs.get("type", "EDGE")
        props = {k: v for k, v in attrs.items() if k != "type" and v is not None}
        prop_str = (
            ", ".join([f"{k}: {json.dumps(v)}" for k, v in props.items()])
            if props
            else ""
        )
        prop_clause = f" {{{prop_str}}}" if prop_str else ""
        lines.append(
            f"MATCH (a:Node {{id: {json.dumps(u)}}}), (b:Node {{id: {json.dumps(v)}}}) "
            f"MERGE (a)-[:{edge_type}{prop_clause}]->(b);"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def export_graph(G: nx.MultiDiGraph, out_dir: Path) -> None:
    """
    Export graph in multiple formats.

    Args:
        G: NetworkX graph.
        out_dir: Output directory.
    """
    # Cypher
    to_cypher(G, out_dir / "graph.cypher")

    # GraphML
    nx.write_graphml(G, str(out_dir / "graph.graphml"))

    # JSON (node-link)
    data = nx.node_link_data(G)
    (out_dir / "graph.json").write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
