#!/usr/bin/env python3
"""
Quick check if ArangoDB is running and accessible.
"""

import os
from dotenv import load_dotenv

load_dotenv()

def check_arango():
    """Check ArangoDB connection."""
    storage_backend = os.getenv("STORAGE_BACKEND", "arango")

    print("=" * 60)
    print("ArangoDB Connection Check")
    print("=" * 60)
    print(f"Storage Backend: {storage_backend}")

    if storage_backend.lower() != "arango":
        print(f"\n✅ Using {storage_backend} storage (no ArangoDB needed)")
        return

    print(f"Host: {os.getenv('ARANGO_HOST', 'localhost')}")
    print(f"Port: {os.getenv('ARANGO_PORT', '8529')}")
    print(f"Database: {os.getenv('ARANGO_DB', 'pdfkg')}")
    print()

    try:
        from pdfkg.db import ArangoDBClient

        print("Connecting to ArangoDB...")
        client = ArangoDBClient()
        db = client.connect()

        print("✅ Connected successfully!")
        print()

        # Show collections
        print("Collections:")
        for collection in db.collections():
            if not collection['name'].startswith('_'):
                count = db.collection(collection['name']).count()
                print(f"  - {collection['name']}: {count} documents")

        # Show PDFs
        pdfs = client.list_pdfs()
        print(f"\nPDFs in database: {len(pdfs)}")
        for pdf in pdfs:
            print(f"  - {pdf['filename']} ({pdf['num_chunks']} chunks)")

        print("\n✅ ArangoDB is ready!")

    except Exception as e:
        print(f"❌ Connection failed!")
        print(f"Error: {e}")
        print()
        print("To fix:")
        print("  1. Start ArangoDB: ./start_arango.sh")
        print("  2. Or switch to file storage in .env:")
        print("     STORAGE_BACKEND=file")
        return False

    return True

if __name__ == "__main__":
    check_arango()
