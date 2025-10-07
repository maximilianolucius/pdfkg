#!/usr/bin/env python3
"""
pdfkg CLI: Build knowledge graphs from technical PDF manuals.

Usage examples:

# Auto-ingest all PDFs from data/input/ (default behavior)
python cli.py

# Process a specific PDF
python cli.py --pdf data/input/your.pdf

# With Gemini visual analysis (processes all PDFs in data/input/)
export GEMINI_API_KEY=YOUR_KEY
python cli.py --use-gemini

# Process specific PDF with Gemini on specific pages
python cli.py --pdf data/input/your.pdf --use-gemini --gemini-pages 1-10,30-40

# Different embedding model
python cli.py --embed-model BAAI/bge-small-en-v1.5
"""

import argparse
import os
from pathlib import Path
import sys
from dotenv import load_dotenv

# Load .env file
load_dotenv()

from pdfkg.storage import get_storage_backend
from pdfkg.pdf_manager import ingest_pdf, auto_ingest_directory


def main():
    parser = argparse.ArgumentParser(
        description="Build knowledge graph from technical PDF manual(s)."
    )
    parser.add_argument(
        "--pdf",
        type=str,
        help="Path to specific PDF file. If not provided, auto-ingests all PDFs from data/input/"
    )
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
    parser.add_argument(
        "--force", action="store_true", help="Force reprocessing even if PDF already exists in cache"
    )

    args = parser.parse_args()

    # Initialize storage backend
    storage = None if args.no_db else get_storage_backend()

    # Setup output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine if processing single PDF or auto-ingesting all
    if args.pdf:
        # Single PDF mode
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"Error: PDF not found at {pdf_path}", file=sys.stderr)
            sys.exit(1)

        # Progress callback for console output
        def print_progress(pct: float, desc: str):
            """Print progress updates to console."""
            if pct < 0.1:
                print(f"{desc}")
            else:
                print(f"[{int(pct * 100):3d}%] {desc}")

        print(f"Processing PDF: {pdf_path.name}")
        if args.force:
            print("  --force enabled: Will reprocess even if cached")
        if args.use_gemini:
            if args.gemini_pages:
                print(f"  Gemini enabled for pages: {args.gemini_pages}")
            else:
                print(f"  Gemini enabled for ALL pages")

        print()

        try:
            # Run unified ingestion pipeline
            result = ingest_pdf(
                pdf_path=pdf_path,
                storage=storage,
                embed_model=args.embed_model,
                max_tokens=args.max_tokens,
                use_gemini=args.use_gemini,
                gemini_pages=args.gemini_pages,
                force_reprocess=args.force,
                save_to_db=not args.no_db,
                save_files=True,
                output_dir=out_dir,
                progress_callback=print_progress,
            )

            # Print summary
            summary = result.summary()
            print("\n=== Pipeline complete ===")

            if result.was_cached:
                print("✓ Loaded from cache")
            else:
                print("✓ Processed successfully")

            print(f"\n📄 Document: {summary['filename']}")
            print(f"   Slug: {summary['pdf_slug']}")
            print(f"\n📊 Statistics:")
            print(f"   Pages: {summary['num_pages']}")
            print(f"   Sections: {summary['num_sections']}")
            print(f"   Text chunks: {summary['num_chunks']}")
            print(f"   Figures: {summary['num_figures']}")
            print(f"   Tables: {summary['num_tables']}")
            print(f"   Cross-references: {summary['num_mentions']} ({summary['num_resolved_mentions']} resolved)")
            print(f"   Graph nodes: {summary['num_graph_nodes']}")
            print(f"   Graph edges: {summary['num_graph_edges']}")
            print(f"   Embedding dimension: {summary['embedding_dim']}")

            if storage and not args.no_db:
                storage_type = os.getenv("STORAGE_BACKEND", "arango")
                print(f"\n💾 Saved to {storage_type.upper()} database")

            print(f"\n📁 File outputs in: {out_dir}")
            print("   - chunks.parquet")
            print("   - mentions.parquet")
            print("   - sections.json")
            print("   - toc.json")
            if args.no_db:
                print("   - index.faiss")
            print("   - graph.cypher")
            print("   - graph.graphml")
            print("   - graph.json")
            print("   - report.md")

        except Exception as e:
            print(f"\n❌ Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)

    else:
        # Auto-ingest all PDFs from data/input/
        input_dir = Path("data/input")

        print("=" * 80)
        print("AUTO-INGEST MODE: Processing all PDFs from data/input/")
        print("=" * 80)

        if args.force:
            print("⚠️  --force flag ignored in auto-ingest mode (only processes new PDFs)")
        if args.use_gemini:
            print("✓ Gemini visual analysis enabled for all PDFs")

        print(f"\nScanning directory: {input_dir}")
        print()

        # Progress callback for auto-ingest
        def auto_progress(pdf_name: str, status: str):
            """Print progress for auto-ingestion."""
            print(f"  [{pdf_name}] {status}")

        try:
            results = auto_ingest_directory(
                input_dir=input_dir,
                storage=storage,
                embed_model=args.embed_model,
                max_tokens=args.max_tokens,
                use_gemini=args.use_gemini,
                save_to_db=not args.no_db,
                save_files=True,
                progress_callback=auto_progress,
            )

            # Print summary
            print("\n" + "=" * 80)
            print("AUTO-INGEST COMPLETE")
            print("=" * 80)

            total = len(results['processed']) + len(results['skipped']) + len(results['failed'])

            if total == 0:
                print("\n⚠️  No PDF files found in data/input/")
                print("   Place PDF files in data/input/ to process them automatically")
            else:
                print(f"\n📊 Summary:")
                print(f"   Total PDFs found: {total}")
                print(f"   ✓ Newly processed: {len(results['processed'])}")
                print(f"   ⊘ Skipped (cached): {len(results['skipped'])}")
                print(f"   ✗ Failed: {len(results['failed'])}")

                if results['processed']:
                    print(f"\n✓ Processed PDFs:")
                    for pdf in results['processed']:
                        print(f"   - {pdf}")

                if results['skipped']:
                    print(f"\n⊘ Skipped PDFs (already in database):")
                    for pdf in results['skipped']:
                        print(f"   - {pdf}")

                if results['failed']:
                    print(f"\n✗ Failed PDFs:")
                    for pdf in results['failed']:
                        print(f"   - {pdf}")

                if storage and not args.no_db:
                    storage_type = os.getenv("STORAGE_BACKEND", "arango")
                    print(f"\n💾 All PDFs saved to {storage_type.upper()} database")

                print(f"\n📁 File outputs in: data/out/<pdf-slug>/")

        except Exception as e:
            print(f"\n❌ Error during auto-ingestion: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
