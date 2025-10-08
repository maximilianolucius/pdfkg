#!/usr/bin/env python3
"""
View Q&A history from the database.

Usage:
    python view_qa_history.py                    # Show last 20 interactions
    python view_qa_history.py --limit 50         # Show last 50
    python view_qa_history.py --pdf my-manual    # Filter by PDF slug
    python view_qa_history.py --llm gemini       # Filter by LLM provider
"""

import argparse
from datetime import datetime
from pdfkg.storage import get_storage_backend


def format_timestamp(iso_timestamp: str) -> str:
    """Format ISO timestamp to readable format."""
    dt = datetime.fromisoformat(iso_timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def print_qa_entry(entry: dict, index: int, verbose: bool = False):
    """Print a single Q&A entry."""
    print(f"\n{'='*80}")
    print(f"[{index}] {format_timestamp(entry['timestamp'])}")
    print(f"{'='*80}")
    print(f"📄 PDF: {entry['pdf_slug']}")
    print(f"🤖 LLM: {entry['llm_provider']} ({entry['llm_model']})")
    print(f"📊 Model: {entry['embed_model']}, Top-K: {entry['top_k']}")
    print(f"⏱️  Response Time: {entry['response_time_ms']:.0f}ms")
    print(f"\n❓ Question:")
    print(f"   {entry['question']}")
    print(f"\n💬 Answer:")
    # Indent answer
    answer_lines = entry['answer'].split('\n')
    for line in answer_lines:
        print(f"   {line}")

    if verbose:
        print(f"\n📚 Sources ({len(entry['sources'])} chunks):")
        for i, src in enumerate(entry['sources'][:5], 1):  # Show first 5
            print(f"   [{i}] Section {src.get('section_id', 'N/A')}, "
                  f"Page {src.get('page', 'N/A')}, "
                  f"Score: {src.get('similarity_score', 0):.3f}")

        related = entry.get('related_items', {})
        if related:
            if related.get('figures'):
                print(f"\n🖼️  Related Figures: {', '.join(related['figures'][:3])}")
            if related.get('tables'):
                print(f"📊 Related Tables: {', '.join(related['tables'][:3])}")
            if related.get('sections'):
                print(f"📖 Related Sections: {', '.join(related['sections'][:3])}")


def main():
    parser = argparse.ArgumentParser(
        description="View Q&A history from the database"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of entries to show (default: 20)",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        help="Filter by PDF slug",
    )
    parser.add_argument(
        "--llm",
        type=str,
        choices=["none", "gemini", "mistral"],
        help="Filter by LLM provider",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed sources and related items",
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export to JSON file",
    )

    args = parser.parse_args()

    # Get storage backend
    try:
        storage = get_storage_backend()
        if not hasattr(storage, 'db_client'):
            print("❌ Error: Q&A history is only available with ArangoDB storage.")
            print("   Set STORAGE_BACKEND=arango in .env and run ./start_arango.sh")
            return 1
    except Exception as e:
        print(f"❌ Error connecting to storage: {e}")
        return 1

    # Get Q&A history
    try:
        history = storage.db_client.get_qa_history(
            pdf_slug=args.pdf,
            limit=args.limit,
            llm_provider=args.llm,
        )
    except Exception as e:
        print(f"❌ Error retrieving Q&A history: {e}")
        return 1

    if not history:
        print("ℹ️  No Q&A history found.")
        if args.pdf:
            print(f"   (filtered by PDF: {args.pdf})")
        if args.llm:
            print(f"   (filtered by LLM: {args.llm})")
        return 0

    # Print summary
    print(f"\n📊 Q&A History Summary")
    print(f"{'='*80}")
    print(f"Total entries: {len(history)}")

    # Count by LLM provider
    llm_counts = {}
    pdf_counts = {}
    for entry in history:
        llm = entry['llm_provider']
        pdf = entry['pdf_slug']
        llm_counts[llm] = llm_counts.get(llm, 0) + 1
        pdf_counts[pdf] = pdf_counts.get(pdf, 0) + 1

    print(f"\nBy LLM Provider:")
    for llm, count in sorted(llm_counts.items()):
        print(f"  - {llm}: {count}")

    print(f"\nBy PDF:")
    for pdf, count in sorted(pdf_counts.items()):
        print(f"  - {pdf}: {count}")

    # Print entries
    for i, entry in enumerate(history, 1):
        print_qa_entry(entry, i, verbose=args.verbose)

    # Export if requested
    if args.export:
        import json
        with open(args.export, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        print(f"\n✅ Exported {len(history)} entries to {args.export}")

    return 0


if __name__ == "__main__":
    exit(main())
