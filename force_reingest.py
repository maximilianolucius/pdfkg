#!/usr/bin/env python3
"""
Force reingest all PDFs from data/input/ into Milvus and ArangoDB.

This script will:
1. List all PDFs in data/input/
2. Delete them from both ArangoDB and Milvus
3. Reingest them using cli.py

Usage:
    python force_reingest.py
"""

import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pdfkg.storage import get_storage_backend
from pdfkg.topology import slugify


def main():
    print("=" * 80)
    print("FORCE REINGEST ALL PDFs TO MILVUS")
    print("=" * 80)

    # Initialize storage
    print("\n🔌 Connecting to storage backend...")
    storage = get_storage_backend()

    # Find all PDFs in data/input/
    input_dir = Path("data/input")
    if not input_dir.exists():
        print(f"❌ Directory not found: {input_dir}")
        return

    pdf_files = list(input_dir.glob("*.pdf"))
    print(f"\n📚 Found {len(pdf_files)} PDF files in {input_dir}")

    if not pdf_files:
        print("No PDFs to process.")
        return

    # Delete existing PDFs from storage
    print("\n🗑️  Deleting existing PDFs from storage and cache...")
    deleted_count = 0

    for pdf_file in pdf_files:
        slug = slugify(pdf_file.stem)

        # Delete cache directory
        cache_dir = Path("data/out") / slug
        if cache_dir.exists():
            import shutil
            print(f"  Deleting cache: {cache_dir}")
            shutil.rmtree(cache_dir)

        # Check if PDF exists in storage
        if storage.db_client.pdf_exists(slug):
            print(f"  Deleting from storage: {pdf_file.name} (slug: {slug})")

            # Delete from ArangoDB
            storage.db_client.delete_pdf(slug)

            # Delete from Milvus
            if storage.milvus_client:
                try:
                    storage.milvus_client.delete_embeddings(slug)
                except Exception as e:
                    print(f"    ⚠️  Milvus delete warning: {e}")

            deleted_count += 1
        else:
            print(f"  Not in storage: {pdf_file.name}")

    print(f"\n✅ Deleted {deleted_count} PDFs from storage")

    # Reingest all PDFs using cli.py
    print("\n" + "=" * 80)
    print("REINGESTING ALL PDFs")
    print("=" * 80)

    for pdf_file in pdf_files:
        print(f"\n📄 Processing: {pdf_file.name}")
        result = subprocess.run(
            [sys.executable, "cli.py", "--pdf", str(pdf_file)],
            cwd=Path(__file__).parent,
        )

        if result.returncode != 0:
            print(f"  ❌ Failed to process {pdf_file.name}")
        else:
            print(f"  ✅ Successfully processed {pdf_file.name}")

    print("\n" + "=" * 80)
    print("REINGEST COMPLETE")
    print("=" * 80)
    print(f"\n✅ Processed {len(pdf_files)} PDFs")
    print("\n💡 Next steps:")
    print("   - Build cross-document graph: python build_cross_document_graph.py")
    print("   - Explore relationships: python explore_cross_document.py")


if __name__ == "__main__":
    main()
