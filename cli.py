#!/usr/bin/env python3
"""
pdfkg CLI: Build knowledge graphs from technical PDF manuals.

Usage examples:

# Basic (no Gemini)
python cli.py --pdf data/input/your.pdf --out data/out

# With Gemini on pages 1-10 and 30-40
export GEMINI_API_KEY=YOUR_KEY
python cli.py --pdf data/input/your.pdf --use-gemini --gemini-pages 1-10,30-40

# Different embedding model
python cli.py --pdf data/input/your.pdf --embed-model BAAI/bge-small-en-v1.5
"""

import argparse
import os
from pathlib import Path
import sys

import faiss
import pandas as pd
import orjson
from dotenv import load_dotenv

# Load .env file
load_dotenv()

from pdfkg.config import Paths
from pdfkg.parse_pdf import load_pdf, extract_pages, extract_toc
from pdfkg.topology import build_section_tree
from pdfkg.chunking import build_chunks
from pdfkg.embeds import embed_chunks, build_faiss_index
from pdfkg.figtables import index_figures_tables
from pdfkg.xrefs import extract_mentions, resolve_mentions
from pdfkg.graph import build_graph, export_graph
from pdfkg.report import generate_report
from pdfkg.storage import get_storage_backend

# Optional Gemini
try:
    from pdfkg.gemini_helpers import gemini_extract_crossrefs, merge_gemini_crossrefs

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def parse_page_ranges(ranges_str: str) -> list[tuple[int, int]]:
    """
    Parse page ranges like '1-10,30-40' into list of (start, end) tuples.

    Args:
        ranges_str: Comma-separated ranges.

    Returns:
        List of (start, end) inclusive tuples.
    """
    ranges = []
    for part in ranges_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            ranges.append((int(start), int(end)))
        else:
            page = int(part)
            ranges.append((page, page))
    return ranges


def main():
    parser = argparse.ArgumentParser(
        description="Build knowledge graph from technical PDF manual."
    )
    parser.add_argument("--pdf", type=str, required=True, help="Path to PDF file")
    parser.add_argument(
        "--out", type=str, default="data/out", help="Output directory for legacy file export (default: data/out)"
    )
    parser.add_argument(
        "--no-db", action="store_true", help="Don't save to database, only export files"
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model name",
    )
    parser.add_argument(
        "--use-gemini", action="store_true", help="Use Gemini for visual cross-ref extraction"
    )
    parser.add_argument(
        "--gemini-pages",
        type=str,
        default="",
        help="Page ranges for Gemini (e.g., 1-10,30-40)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=500, help="Max tokens per chunk"
    )

    args = parser.parse_args()

    # Initialize storage backend
    storage = None if args.no_db else get_storage_backend()

    # Paths
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}", file=sys.stderr)
        sys.exit(1)

    # Create slug from filename
    original_filename = pdf_path.name
    pdf_slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in pdf_path.stem).lower()

    # Save to data/input/
    input_dir = Path("data/input")
    input_dir.mkdir(parents=True, exist_ok=True)
    if not (input_dir / original_filename).exists():
        import shutil
        shutil.copy(pdf_path, input_dir / original_filename)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading PDF: {pdf_path}")
    doc = load_pdf(pdf_path)

    print("Extracting pages...")
    pages = extract_pages(doc)
    print(f"  {len(pages)} pages extracted")

    print("Extracting ToC...")
    toc = extract_toc(doc)
    print(f"  {len(toc)} ToC entries")

    print("Building section tree...")
    sections = build_section_tree(toc)
    print(f"  {len(sections)} sections")

    print("Building chunks...")
    chunks = build_chunks(pages, sections, max_tokens=args.max_tokens)
    print(f"  {len(chunks)} chunks created")

    print(f"Embedding chunks with {args.embed_model}...")
    embeddings = embed_chunks(chunks, model_name=args.embed_model)
    print(f"  Embeddings shape: {embeddings.shape}")

    print("Building FAISS index...")
    if storage and not args.no_db:
        storage.save_embeddings(pdf_slug, embeddings)
        print(f"  Embeddings saved to database")
    else:
        index = build_faiss_index(embeddings)
        faiss.write_index(index, str(out_dir / "index.faiss"))
        print(f"  Index saved: {out_dir / 'index.faiss'}")

    print("Indexing figures and tables...")
    figures, tables = index_figures_tables(pages)
    print(f"  Figures: {len(figures)}, Tables: {len(tables)}")

    print("Extracting cross-reference mentions...")
    all_mentions = []
    for chunk in chunks:
        all_mentions.extend(extract_mentions(chunk))
    print(f"  {len(all_mentions)} mentions found")

    print("Resolving mentions...")
    all_mentions = resolve_mentions(
        all_mentions, sections, figures, tables, n_pages=len(pages)
    )
    resolved_count = sum(1 for m in all_mentions if m.target_id)
    print(f"  Resolved: {resolved_count}/{len(all_mentions)}")

    # Optional Gemini
    if args.use_gemini:
        if not GEMINI_AVAILABLE:
            print("Warning: google-generativeai not installed, skipping Gemini", file=sys.stderr)
        elif not os.getenv("GEMINI_API_KEY"):
            print("Warning: GEMINI_API_KEY not set, skipping Gemini", file=sys.stderr)
        else:
            print("Using Gemini for visual cross-ref extraction...")
            if not args.gemini_pages:
                print("  No --gemini-pages specified, skipping Gemini")
            else:
                page_ranges = parse_page_ranges(args.gemini_pages)
                gemini_results = {"cross_references": []}
                for start, end in page_ranges:
                    print(f"  Processing pages {start}-{end} with Gemini...")
                    result = gemini_extract_crossrefs(pdf_path, start, end)
                    gemini_results["cross_references"].extend(
                        result.get("cross_references", [])
                    )
                print(f"  Gemini found {len(gemini_results['cross_references'])} references")
                print("  Merging Gemini results...")
                all_mentions = merge_gemini_crossrefs(
                    all_mentions, gemini_results, sections, figures, tables
                )
                resolved_count = sum(1 for m in all_mentions if m.target_id)
                print(f"  Resolved after Gemini: {resolved_count}/{len(all_mentions)}")

    print("Building knowledge graph...")
    graph = build_graph(
        doc_id="document",
        pages=pages,
        sections=sections,
        chunks=chunks,
        mentions=all_mentions,
        figures=figures,
        tables=tables,
    )
    print(
        f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}"
    )

    print("Exporting graph...")
    export_graph(graph, out_dir)
    print(f"  graph.cypher, graph.graphml, graph.json saved")

    print("Saving artifacts...")

    # Save to database
    if storage and not args.no_db:
        # Save chunks
        chunks_data = [
            {"chunk_id": c.id, "section_id": c.section_id, "page": c.page, "text": c.text}
            for c in chunks
        ]
        storage.save_chunks(pdf_slug, chunks_data)
        print(f"  Saved {len(chunks_data)} chunks to database")

        # Save graph
        if hasattr(storage, 'save_graph'):
            nodes = []
            for node_id, attrs in graph.nodes(data=True):
                node_doc = {"node_id": node_id, "type": attrs.get("type", "Unknown"), "label": attrs.get("label", "")}
                for k, v in attrs.items():
                    if k not in ["type", "label"] and v is not None:
                        node_doc[k] = v
                nodes.append(node_doc)

            edges = []
            for u, v, attrs in graph.edges(data=True):
                edge_doc = {"from_id": u, "to_id": v, "type": attrs.get("type", "EDGE")}
                for k, v in attrs.items():
                    if k not in ["type"] and v is not None:
                        edge_doc[k] = v
                edges.append(edge_doc)

            storage.save_graph(pdf_slug, nodes, edges)
            print(f"  Saved graph ({len(nodes)} nodes, {len(edges)} edges) to database")

        # Register PDF FIRST (required before saving metadata)
        storage.save_pdf_metadata(
            slug=pdf_slug,
            filename=original_filename,
            num_pages=len(pages),
            num_chunks=len(chunks),
            num_sections=len(sections),
            num_figures=len(figures),
            num_tables=len(tables),
        )
        print(f"  Registered PDF in database")

        # Save metadata (AFTER PDF is registered)
        storage.save_metadata(pdf_slug, "sections", sections)
        storage.save_metadata(pdf_slug, "toc", toc)
        storage.save_metadata(pdf_slug, "mentions", [
            {
                "source_chunk_id": m.source_chunk_id,
                "kind": m.kind,
                "raw_text": m.raw_text,
                "target_hint": m.target_hint,
                "target_id": m.target_id,
            }
            for m in all_mentions
        ])
        storage.save_metadata(pdf_slug, "figures", figures)
        storage.save_metadata(pdf_slug, "tables", tables)
        print(f"  Saved metadata to database")

    # Also save to files for compatibility
    chunks_df = pd.DataFrame(
        [{"id": c.id, "section_id": c.section_id, "page": c.page, "text": c.text} for c in chunks]
    )
    chunks_df.to_parquet(out_dir / "chunks.parquet", index=False)

    mentions_df = pd.DataFrame(
        [
            {
                "source_chunk_id": m.source_chunk_id,
                "kind": m.kind,
                "raw_text": m.raw_text,
                "target_hint": m.target_hint,
                "target_id": m.target_id,
            }
            for m in all_mentions
        ]
    )
    mentions_df.to_parquet(out_dir / "mentions.parquet", index=False)

    (out_dir / "sections.json").write_bytes(
        orjson.dumps(sections, option=orjson.OPT_INDENT_2)
    )
    (out_dir / "toc.json").write_bytes(orjson.dumps(toc, option=orjson.OPT_INDENT_2))

    print("Generating report...")
    generate_report(
        out_dir / "report.md", sections, chunks, all_mentions, figures, tables
    )

    print("\n=== Pipeline complete ===")
    if storage and not args.no_db:
        storage_type = os.getenv("STORAGE_BACKEND", "arango")
        print(f"PDF saved to {storage_type.upper()} database:")
        print(f"  - Slug: {pdf_slug}")
        print(f"  - Chunks: {len(chunks)}")
        print(f"  - Graph nodes: {graph.number_of_nodes()}")
        print(f"  - Graph edges: {graph.number_of_edges()}")
    print(f"\nLegacy file outputs in: {out_dir}")
    print("  - chunks.parquet")
    print("  - mentions.parquet")
    print("  - sections.json")
    print("  - toc.json")
    print("  - index.faiss (if --no-db)")
    print("  - graph.cypher")
    print("  - graph.graphml")
    print("  - graph.json")
    print("  - report.md")


if __name__ == "__main__":
    main()
