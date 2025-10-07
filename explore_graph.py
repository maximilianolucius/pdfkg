#!/usr/bin/env python3
"""
Script to explore graphs in ArangoDB.

Usage:
    python explore_graph.py                    # List all PDFs
    python explore_graph.py --pdf manual       # Explore specific PDF
    python explore_graph.py --visualize manual # Export to NetworkX/GraphML
"""

import argparse
from pdfkg.storage import get_storage_backend
import networkx as nx
from pathlib import Path


def list_pdfs(storage):
    """List all PDFs in database."""
    print("\n" + "="*80)
    print("📚 Available PDFs in ArangoDB")
    print("="*80)

    pdfs = storage.list_pdfs()

    if not pdfs:
        print("⚠️  No PDFs found in database")
        return

    for i, pdf in enumerate(pdfs, 1):
        print(f"\n{i}. {pdf['filename']}")
        print(f"   Slug: {pdf['slug']}")
        print(f"   Pages: {pdf['num_pages']}")
        print(f"   Chunks: {pdf['num_chunks']}")
        print(f"   Sections: {pdf['num_sections']}")
        print(f"   Figures: {pdf.get('num_figures', 0)}")
        print(f"   Tables: {pdf.get('num_tables', 0)}")
        print(f"   Processed: {pdf['processed_date'][:10]}")


def explore_graph(storage, pdf_slug):
    """Explore graph structure for a PDF."""
    print(f"\n" + "="*80)
    print(f"🔍 Exploring graph for: {pdf_slug}")
    print("="*80)

    # Get PDF info
    pdf_info = storage.get_pdf_metadata(pdf_slug)
    if not pdf_info:
        print(f"❌ PDF not found: {pdf_slug}")
        return

    print(f"\n📄 PDF: {pdf_info['filename']}")

    # Get graph
    if not hasattr(storage, 'get_graph'):
        print("⚠️  Storage backend doesn't support graph operations")
        return

    nodes, edges = storage.get_graph(pdf_slug)

    print(f"\n📊 Graph Statistics:")
    print(f"   Total nodes: {len(nodes)}")
    print(f"   Total edges: {len(edges)}")

    # Count node types
    node_types = {}
    for node in nodes:
        ntype = node.get('type', 'Unknown')
        node_types[ntype] = node_types.get(ntype, 0) + 1

    print(f"\n📦 Node Types:")
    for ntype, count in sorted(node_types.items()):
        print(f"   - {ntype}: {count}")

    # Count edge types
    edge_types = {}
    for edge in edges:
        etype = edge.get('type', 'Unknown')
        edge_types[etype] = edge_types.get(etype, 0) + 1

    print(f"\n🔗 Edge Types:")
    for etype, count in sorted(edge_types.items()):
        print(f"   - {etype}: {count}")

    # Sample nodes
    print(f"\n📝 Sample Nodes (first 5):")
    for i, node in enumerate(nodes[:5], 1):
        print(f"\n   {i}. {node.get('node_id', 'N/A')}")
        print(f"      Type: {node.get('type', 'N/A')}")
        print(f"      Label: {node.get('label', 'N/A')}")

    # Sample edges
    print(f"\n🔗 Sample Edges (first 5):")
    for i, edge in enumerate(edges[:5], 1):
        print(f"\n   {i}. {edge.get('from_id', 'N/A')} → {edge.get('to_id', 'N/A')}")
        print(f"      Type: {edge.get('type', 'N/A')}")


def visualize_graph(storage, pdf_slug, output_dir=Path("data/graph_exports")):
    """Export graph to NetworkX and save visualizations."""
    print(f"\n" + "="*80)
    print(f"🎨 Visualizing graph for: {pdf_slug}")
    print("="*80)

    # Load graph
    if hasattr(storage, 'load_graph'):
        graph = storage.load_graph(pdf_slug)
    else:
        nodes, edges = storage.get_graph(pdf_slug)

        # Reconstruct NetworkX graph
        graph = nx.MultiDiGraph()
        for node in nodes:
            node_id = node.get('node_id')
            graph.add_node(node_id, **{k: v for k, v in node.items() if k != 'node_id'})

        for edge in edges:
            from_id = edge.get('from_id')
            to_id = edge.get('to_id')
            graph.add_edge(from_id, to_id, **{k: v for k, v in edge.items()
                                               if k not in ['from_id', 'to_id', '_from', '_to']})

    print(f"\n📊 Graph loaded:")
    print(f"   Nodes: {graph.number_of_nodes()}")
    print(f"   Edges: {graph.number_of_edges()}")

    # Create output directory
    output_dir = Path(output_dir) / pdf_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export to various formats
    print(f"\n💾 Exporting to: {output_dir}")

    # 1. GraphML (for Gephi, Cytoscape, etc.)
    graphml_path = output_dir / "graph.graphml"
    nx.write_graphml(graph, graphml_path)
    print(f"   ✓ GraphML: {graphml_path}")

    # 2. JSON
    from networkx.readwrite import json_graph
    import json

    graph_data = json_graph.node_link_data(graph)
    json_path = output_dir / "graph.json"
    with open(json_path, 'w') as f:
        json.dump(graph_data, f, indent=2)
    print(f"   ✓ JSON: {json_path}")

    # 3. GML (for other tools)
    gml_path = output_dir / "graph.gml"
    nx.write_gml(graph, gml_path)
    print(f"   ✓ GML: {gml_path}")

    # 4. Adjacency list
    adj_path = output_dir / "adjacency.txt"
    nx.write_adjlist(graph, adj_path)
    print(f"   ✓ Adjacency List: {adj_path}")

    # 5. Graph statistics
    stats_path = output_dir / "statistics.txt"
    with open(stats_path, 'w') as f:
        f.write(f"Graph Statistics for: {pdf_slug}\n")
        f.write(f"="*80 + "\n\n")
        f.write(f"Nodes: {graph.number_of_nodes()}\n")
        f.write(f"Edges: {graph.number_of_edges()}\n")
        f.write(f"Density: {nx.density(graph):.4f}\n")

        # Node types
        f.write(f"\nNode Types:\n")
        node_types = {}
        for node, attrs in graph.nodes(data=True):
            ntype = attrs.get('type', 'Unknown')
            node_types[ntype] = node_types.get(ntype, 0) + 1
        for ntype, count in sorted(node_types.items()):
            f.write(f"  - {ntype}: {count}\n")

        # Edge types
        f.write(f"\nEdge Types:\n")
        edge_types = {}
        for u, v, attrs in graph.edges(data=True):
            etype = attrs.get('type', 'Unknown')
            edge_types[etype] = edge_types.get(etype, 0) + 1
        for etype, count in sorted(edge_types.items()):
            f.write(f"  - {etype}: {count}\n")

    print(f"   ✓ Statistics: {stats_path}")

    print(f"\n✅ Graph exported successfully!")
    print(f"\nYou can now:")
    print(f"   - Open {graphml_path} in Gephi")
    print(f"   - Open {graphml_path} in Cytoscape")
    print(f"   - Import {json_path} into D3.js")
    print(f"   - View {stats_path} for statistics")


def main():
    parser = argparse.ArgumentParser(
        description="Explore graphs in ArangoDB"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        help="PDF slug to explore"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Export graph for visualization"
    )

    args = parser.parse_args()

    # Initialize storage
    storage = get_storage_backend()

    if not args.pdf:
        # List all PDFs
        list_pdfs(storage)
    else:
        # Explore specific PDF
        explore_graph(storage, args.pdf)

        if args.visualize:
            visualize_graph(storage, args.pdf)


if __name__ == "__main__":
    main()
