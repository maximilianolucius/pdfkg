#!/usr/bin/env python3
"""
Migrate file-based storage to ArangoDB.
"""

import json
from pathlib import Path

import pandas as pd
import orjson
import networkx as nx

from pdfkg.db import ArangoDBClient
from dotenv import load_dotenv

load_dotenv()


def migrate_to_arango():
    """Migrate all PDFs from file storage to ArangoDB."""
    # Connect to ArangoDB
    print("Connecting to ArangoDB...")
    client = ArangoDBClient()
    client.connect()
    print("✅ Connected to ArangoDB")

    # Load registry
    registry_file = Path("data/out/processed_pdfs.json")
    if not registry_file.exists():
        print("❌ No processed_pdfs.json found. Nothing to migrate.")
        return

    with open(registry_file, "r") as f:
        registry = json.load(f)

    print(f"\nFound {len(registry)} PDFs to migrate\n")

    for slug, pdf_info in registry.items():
        print(f"Migrating: {pdf_info['filename']} (slug: {slug})")

        try:
            # 1. Register PDF
            client.register_pdf(
                slug=slug,
                filename=pdf_info["filename"],
                num_pages=pdf_info["num_pages"],
                num_chunks=pdf_info["num_chunks"],
                num_sections=pdf_info["num_sections"],
            )
            print("  ✅ PDF metadata saved")

            # 2. Load and save chunks
            out_dir = Path(pdf_info["output_dir"])
            chunks_path = out_dir / "chunks.parquet"
            if chunks_path.exists():
                chunks_df = pd.read_parquet(chunks_path)
                chunks = []
                for _, row in chunks_df.iterrows():
                    chunks.append(
                        {
                            "chunk_id": row["id"],
                            "section_id": row["section_id"],
                            "page": int(row["page"]),
                            "text": row["text"],
                        }
                    )
                client.save_chunks(slug, chunks)
                print(f"  ✅ Saved {len(chunks)} chunks")

            # 3. Load and save graph
            graph_path = out_dir / "graph.json"
            if graph_path.exists():
                with open(graph_path, "rb") as f:
                    graph_data = orjson.loads(f.read())

                G = nx.node_link_graph(graph_data)

                # Convert to ArangoDB format
                nodes = []
                for node_id, attrs in G.nodes(data=True):
                    node_doc = {
                        "node_id": node_id,
                        "type": attrs.get("type", "Unknown"),
                        "label": attrs.get("label", ""),
                    }
                    # Add all other attributes
                    for k, v in attrs.items():
                        if k not in ["type", "label"] and v is not None:
                            node_doc[k] = v
                    nodes.append(node_doc)

                edges = []
                for u, v, attrs in G.edges(data=True):
                    edge_doc = {
                        "from_id": u,
                        "to_id": v,
                        "type": attrs.get("type", "EDGE"),
                    }
                    # Add all other attributes
                    for k, v in attrs.items():
                        if k not in ["type"] and v is not None:
                            edge_doc[k] = v
                    edges.append(edge_doc)

                client.save_graph(slug, nodes, edges)
                print(f"  ✅ Saved graph ({len(nodes)} nodes, {len(edges)} edges)")

            # 4. Save metadata (sections, toc, mentions)
            for metadata_key, filename in [
                ("sections", "sections.json"),
                ("toc", "toc.json"),
                ("mentions", "mentions.parquet"),
            ]:
                file_path = out_dir / filename
                if file_path.exists():
                    if filename.endswith(".json"):
                        with open(file_path, "rb") as f:
                            data = orjson.loads(f.read())
                    elif filename.endswith(".parquet"):
                        df = pd.read_parquet(file_path)
                        data = df.to_dict("records")
                    else:
                        continue

                    client.save_metadata(slug, metadata_key, data)
                    print(f"  ✅ Saved {metadata_key}")

            print(f"  ✅ Migration complete for {slug}\n")

        except Exception as e:
            print(f"  ❌ Error migrating {slug}: {e}\n")
            continue

    print("=" * 80)
    print("✅ Migration complete!")
    print(f"   Migrated {len(registry)} PDFs to ArangoDB")
    print(f"\nTo use ArangoDB, add to .env:")
    print(f"   STORAGE_BACKEND=arango")
    print(f"\nFAISS indexes are kept in data/faiss_indexes/")
    print("=" * 80)


if __name__ == "__main__":
    migrate_to_arango()
