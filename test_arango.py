#!/usr/bin/env python3
"""
Test ArangoDB connection and setup.
"""

import os
from dotenv import load_dotenv
load_dotenv()

from pdfkg.db import ArangoDBClient

def test_connection():
    """Test ArangoDB connection."""
    print("Testing ArangoDB connection...")
    print(f"Host: {os.getenv('ARANGO_HOST', 'localhost')}")
    print(f"Port: {os.getenv('ARANGO_PORT', '8529')}")
    print(f"Database: {os.getenv('ARANGO_DB', 'pdfkg')}")
    print()

    try:
        client = ArangoDBClient()
        db = client.connect()
        print("✅ Connected to ArangoDB successfully!")
        print()

        # List collections
        print("Collections:")
        for collection in db.collections():
            if not collection['name'].startswith('_'):
                count = db.collection(collection['name']).count()
                print(f"  - {collection['name']}: {count} documents")

        print()
        print("✅ ArangoDB is ready to use!")

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print()
        print("Make sure ArangoDB is running:")
        print("  ./start_arango.sh")
        print("  or")
        print("  docker-compose up -d")

if __name__ == "__main__":
    test_connection()
