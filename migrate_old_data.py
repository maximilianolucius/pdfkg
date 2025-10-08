#!/usr/bin/env python3
"""
Migrate old single-PDF data structure to new multi-PDF structure.

Run this if you have existing data in data/out/ from before the multi-PDF update.
"""

import shutil
from pathlib import Path
from datetime import datetime

from pdfkg.pdf_manager import PDFManager


def migrate_old_data():
    """Migrate old data to new structure."""
    old_out_dir = Path("data/out")

    # Check if old structure exists
    old_files = [
        "chunks.parquet",
        "index.faiss",
        "graph.json",
        "sections.json",
    ]

    has_old_data = all((old_out_dir / f).exists() for f in old_files)

    if not has_old_data:
        print("No old data found. Nothing to migrate.")
        return

    # Ask for PDF name
    print("Found old data structure. Migrating to multi-PDF format...")
    pdf_name = input("Enter the PDF filename (e.g., manual.pdf): ").strip()

    if not pdf_name:
        print("No filename provided. Aborting.")
        return

    # Initialize PDF manager
    pdf_manager = PDFManager()
    slug = pdf_manager.get_pdf_slug(pdf_name)

    # Create new directory
    new_dir = old_out_dir / slug
    new_dir.mkdir(exist_ok=True)

    print(f"Migrating to: {new_dir}")

    # Move files
    files_to_move = [
        "chunks.parquet",
        "index.faiss",
        "graph.cypher",
        "graph.graphml",
        "graph.json",
        "mentions.parquet",
        "report.md",
        "sections.json",
        "toc.json",
    ]

    for file in files_to_move:
        src = old_out_dir / file
        if src.exists():
            dst = new_dir / file
            shutil.move(str(src), str(dst))
            print(f"  Moved: {file}")

    # Try to get stats from chunks
    try:
        import pandas as pd
        chunks_df = pd.read_parquet(new_dir / "chunks.parquet")
        num_chunks = len(chunks_df)

        # Get sections count
        import orjson
        with open(new_dir / "sections.json", "rb") as f:
            sections = orjson.loads(f.read())
        num_sections = len(sections)

        # Estimate pages from chunks
        num_pages = chunks_df['page'].max() if 'page' in chunks_df.columns else 0
    except:
        num_chunks = 0
        num_sections = 0
        num_pages = 0

    # Register PDF
    pdf_manager.register_pdf(
        filename=pdf_name,
        num_pages=num_pages,
        num_chunks=num_chunks,
        num_sections=num_sections,
    )

    print(f"\n✅ Migration complete!")
    print(f"   PDF: {pdf_name}")
    print(f"   Slug: {slug}")
    print(f"   Directory: {new_dir}")
    print(f"\nYou can now run the app: python app.py")


if __name__ == "__main__":
    migrate_old_data()
