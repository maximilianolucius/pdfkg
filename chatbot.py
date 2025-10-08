#!/usr/bin/env python3
"""
Interactive chatbot for querying the PDF knowledge graph.

Usage:
    python chatbot.py
    python chatbot.py --llm-provider gemini
    python chatbot.py --llm-provider mistral
    python chatbot.py --pdf my-manual.pdf --llm-provider gemini
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Fix for macOS segfault - must be before any torch/transformers imports
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Fix FAISS threading issues on macOS
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from pdfkg.query import answer_question
from pdfkg.storage import get_storage_backend

# Load .env file
load_dotenv()


def print_answer(result: dict, verbose: bool = False) -> None:
    """Print formatted answer."""
    print("\n" + "=" * 80)
    print(f"Q: {result['question']}")
    print("=" * 80)
    print(f"\n{result['answer']}\n")

    if verbose:
        print("\n--- Sources ---")
        for i, source in enumerate(result["sources"], 1):
            print(
                f"[{i}] Section {source['section_id']}, Page {source['page']} (score: {source['similarity_score']:.3f})"
            )

        related = result["related"]
        if related["figures"]:
            print(f"\nRelated Figures: {', '.join(related['figures'])}")
        if related["tables"]:
            print(f"Related Tables: {', '.join(related['tables'])}")
        if related["sections"]:
            print(f"Related Sections: {', '.join(related['sections'])}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Interactive chatbot for PDF Q&A.")
    parser.add_argument(
        "--pdf",
        type=str,
        help="PDF slug or filename to query (optional, will prompt if not specified)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/out",
        help="[DEPRECATED] Output directory - now uses database by default",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model name",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of chunks to retrieve"
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        choices=["none", "gemini", "mistral"],
        default="none",
        help="LLM provider to use for answer generation (none, gemini, or mistral)",
    )
    parser.add_argument(
        "--use-gemini",
        action="store_true",
        help="[DEPRECATED] Use --llm-provider gemini instead",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed sources and related nodes"
    )
    parser.add_argument(
        "--question", type=str, help="Ask a single question (non-interactive)"
    )

    args = parser.parse_args()

    # Handle deprecated --use-gemini flag
    if args.use_gemini:
        print("Warning: --use-gemini is deprecated. Use --llm-provider gemini instead.")
        args.llm_provider = "gemini"

    # Initialize storage backend
    storage = get_storage_backend()

    # List available PDFs
    pdf_list = storage.list_pdfs()
    if not pdf_list:
        print("Error: No PDFs found in database.", file=sys.stderr)
        print("Process a PDF first:", file=sys.stderr)
        print("  python cli.py --pdf <your.pdf>", file=sys.stderr)
        print("  or", file=sys.stderr)
        print("  python app.py  (web interface)", file=sys.stderr)
        sys.exit(1)

    # Select PDF
    if args.pdf:
        # Try to find PDF by slug or filename
        selected_pdf = None
        for pdf in pdf_list:
            if pdf['slug'] == args.pdf or pdf['filename'] == args.pdf:
                selected_pdf = pdf['slug']
                break
        if not selected_pdf:
            print(f"Error: PDF not found: {args.pdf}", file=sys.stderr)
            print(f"Available PDFs:", file=sys.stderr)
            for pdf in pdf_list:
                print(f"  - {pdf['filename']} (slug: {pdf['slug']})", file=sys.stderr)
            sys.exit(1)
    else:
        # Show list and prompt
        print("Available PDFs:")
        for i, pdf in enumerate(pdf_list, 1):
            print(f"  {i}. {pdf['filename']} ({pdf['num_chunks']} chunks, {pdf['num_pages']} pages)")

        if len(pdf_list) == 1:
            selected_pdf = pdf_list[0]['slug']
            print(f"\nUsing: {pdf_list[0]['filename']}")
        else:
            try:
                choice = int(input("\nSelect PDF number: ")) - 1
                if 0 <= choice < len(pdf_list):
                    selected_pdf = pdf_list[choice]['slug']
                else:
                    print("Invalid choice.", file=sys.stderr)
                    sys.exit(1)
            except (ValueError, KeyboardInterrupt):
                print("\nCancelled.", file=sys.stderr)
                sys.exit(1)

    # Check LLM provider availability
    llm_status = "none"
    if args.llm_provider == "gemini":
        if os.getenv("GEMINI_API_KEY"):
            gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            llm_status = f"Gemini ({gemini_model})"
        else:
            print("Warning: Gemini selected but GEMINI_API_KEY not found in environment")
            print("Set it in .env file or export GEMINI_API_KEY=your_key")
            llm_status = "Gemini (disabled - no API key)"
    elif args.llm_provider == "mistral":
        if os.getenv("MISTRAL_API_KEY"):
            mistral_model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
            llm_status = f"Mistral ({mistral_model})"
        else:
            print("Warning: Mistral selected but MISTRAL_API_KEY not found in environment")
            print("Set it in .env file or export MISTRAL_API_KEY=your_key")
            llm_status = "Mistral (disabled - no API key)"

    # Get PDF info
    pdf_info = storage.get_pdf_metadata(selected_pdf)

    print("=" * 80)
    print("PDF Knowledge Graph Chatbot")
    print("=" * 80)
    print(f"PDF: {pdf_info['filename']}")
    print(f"Pages: {pdf_info['num_pages']}, Chunks: {pdf_info['num_chunks']}")
    print(f"Storage: {os.getenv('STORAGE_BACKEND', 'arango').upper()}")
    print(f"Embedding Model: {args.embed_model}")
    print(f"LLM Provider: {llm_status}")
    print("=" * 80)

    # Single question mode
    if args.question:
        result = answer_question(
            args.question,
            selected_pdf,
            model_name=args.embed_model,
            top_k=args.top_k,
            llm_provider=args.llm_provider,
            storage=storage,
        )
        print_answer(result, verbose=args.verbose)
        return

    # Interactive mode
    print("\nEnter your questions (type 'quit' or 'exit' to end)\n")

    while True:
        try:
            question = input("Question: ").strip()
            if not question:
                continue
            if question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            result = answer_question(
                question,
                selected_pdf,
                model_name=args.embed_model,
                top_k=args.top_k,
                llm_provider=args.llm_provider,
                storage=storage,
            )
            print_answer(result, verbose=args.verbose)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
