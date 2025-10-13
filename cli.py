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

# AAS (Asset Administration Shell) - Phase 1: Classify
export GEMINI_API_KEY=YOUR_KEY
python cli.py --classify-aas --llm-provider gemini

# AAS Phase 2: Extract structured data (requires Phase 1 first)
python cli.py --extract-aas --llm-provider gemini

# AAS Phase 3: Validate and complete data (requires Phase 2 first)
python cli.py --validate-aas --llm-provider gemini

# AAS Phase 4: Generate AAS v5.0 XML (requires Phase 3 first)
python cli.py --generate-aas-xml --llm-provider gemini

# Run all 4 phases: classify + extract + validate + generate XML
python cli.py --classify-aas --extract-aas --validate-aas --generate-aas-xml --llm-provider gemini

# Full pipeline: auto-ingest + relationships + AAS (all 4 phases)
python cli.py --classify-aas --extract-aas --validate-aas --generate-aas-xml
"""

# Fix for macOS multiprocessing segmentation fault
import multiprocessing
import sys
if sys.platform == "darwin":  # macOS
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

import argparse
import os

# Fix FAISS threading issues on macOS
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

from pdfkg.storage import get_storage_backend
from pdfkg.pdf_manager import ingest_pdf, auto_ingest_directory
from pdfkg.cross_document import CrossDocumentAnalyzer, cross_doc_ref_to_dict, semantic_link_to_dict, version_relation_to_dict
from pdfkg.ner import TechnicalNER, extract_entities_from_chunks, entity_to_dict
from pdfkg.aas_classifier import classify_pdfs_to_aas
from pdfkg.aas_extractor import extract_aas_data
from pdfkg.aas_validator import validate_aas_data
from pdfkg.aas_xml_generator import generate_aas_xml


def build_cross_document_relationships(storage, similarity_threshold: float = 0.85, top_k: int = 10):
    """
    Build cross-document relationships for all PDFs in the database.

    Executes 3 phases:
    - Phase 1: Cross-doc refs, Entities, Versioning
    - Phase 2: Semantic similarity, Topic clustering
    - Phase 3: Citation network

    Args:
        storage: Storage backend
        similarity_threshold: Minimum similarity for semantic links
        top_k: Number of similar chunks to find
    """
    print("\n" + "=" * 80)
    print("BUILDING CROSS-DOCUMENT RELATIONSHIPS")
    print("=" * 80)

    # Get all PDFs
    all_pdfs = storage.list_pdfs()

    if len(all_pdfs) < 2:
        print(f"\n⊘ Skipping relationship building: Need at least 2 PDFs (found {len(all_pdfs)})")
        return

    print(f"\n📚 Found {len(all_pdfs)} PDFs in database")

    # Get Milvus client if available
    milvus_client = None
    if hasattr(storage, 'milvus_client') and storage.milvus_client:
        milvus_client = storage.milvus_client
        print("✓ Milvus client available for semantic search")
    else:
        print("⚠️  Milvus not available - will skip semantic similarity and topic clustering")

    # Initialize analyzer
    analyzer = CrossDocumentAnalyzer(storage, milvus_client)

    # === PHASE 1: MVP ===
    print("\n" + "-" * 80)
    print("Phase 1: Cross-doc refs, Entities, Versioning")
    print("-" * 80)

    # 1.1: Cross-document references
    print("\n📎 Step 1.1: Extracting cross-document references...")
    all_refs = []
    for pdf in all_pdfs:
        slug = pdf['slug']
        refs = analyzer.extract_cross_doc_refs(slug)
        resolved_refs = analyzer.resolve_cross_doc_refs(refs)
        all_refs.extend(resolved_refs)
        if refs:
            print(f"  {pdf['filename']}: {len(refs)} refs ({sum(1 for r in resolved_refs if r.target_pdf)} resolved)")

    if all_refs:
        storage.db_client.save_metadata('__global__', 'cross_doc_refs', [cross_doc_ref_to_dict(r) for r in all_refs])
        print(f"  ✓ Saved {len(all_refs)} cross-document references")

    # 1.2: Named entity extraction
    print("\n🏷️  Step 1.2: Extracting named entities...")
    all_entities = {}
    ner = TechnicalNER()

    for pdf in all_pdfs:
        slug = pdf['slug']
        chunks = storage.get_chunks(slug)
        entity_dict = extract_entities_from_chunks(chunks, include_products=True)
        total_entities = sum(len(entities) for entities in entity_dict.values())
        all_entities[slug] = entity_dict
        if total_entities > 0:
            print(f"  {pdf['filename']}: {total_entities} entities")

    for slug, entity_dict in all_entities.items():
        serializable = {}
        for chunk_id, entities in entity_dict.items():
            serializable[chunk_id] = [entity_to_dict(e) for e in entities]
        storage.db_client.save_metadata(slug, 'extracted_entities', serializable)
    print(f"  ✓ Saved entities for {len(all_entities)} PDFs")

    # 1.3: Document versioning
    print("\n📅 Step 1.3: Detecting version relationships...")
    version_rels = analyzer.detect_version_relationships(all_pdfs)
    if version_rels:
        storage.db_client.save_metadata('__global__', 'version_relations', [version_relation_to_dict(r) for r in version_rels])
        print(f"  ✓ Found {len(version_rels)} version relationships")
    else:
        print(f"  ⊘ No version relationships found")

    # === PHASE 2: Expansion ===
    if milvus_client:
        print("\n" + "-" * 80)
        print("Phase 2: Semantic Similarity & Topic Clustering")
        print("-" * 80)

        # 2.1: Semantic similarity
        print(f"\n🔍 Step 2.1: Finding semantic similarities (threshold={similarity_threshold}, top_k={top_k})...")
        all_semantic_links = []
        for pdf in all_pdfs:
            slug = pdf['slug']
            links = analyzer.find_semantic_similarities(slug, threshold=similarity_threshold, top_k=top_k)
            all_semantic_links.extend(links)
            if links:
                print(f"  {pdf['filename']}: {len(links)} links")

        if all_semantic_links:
            storage.db_client.save_metadata('__global__', 'semantic_links', [semantic_link_to_dict(l) for l in all_semantic_links])
            print(f"  ✓ Saved {len(all_semantic_links)} semantic links")

        # 2.2: Topic clustering
        print("\n📊 Step 2.2: Clustering documents by topics...")
        n_topics = min(10, max(2, len(all_pdfs) // 2))
        topics_result = analyzer.cluster_documents_by_topic(all_pdfs, n_topics=n_topics)

        if topics_result:
            storage.db_client.save_metadata('__global__', 'topics', topics_result)
            print(f"  ✓ Found {topics_result['n_topics']} topics (silhouette: {topics_result['silhouette_score']:.3f})")

    # === PHASE 3: Advanced ===
    print("\n" + "-" * 80)
    print("Phase 3: Citation Network")
    print("-" * 80)

    print("\n📚 Building citation network...")
    citations = analyzer.build_citation_network(all_pdfs)

    if citations:
        storage.db_client.save_metadata('__global__', 'citation_network', citations)
        print(f"  ✓ Saved {len(citations)} citation relationships")
    else:
        print(f"  ⊘ No citations found")

    print("\n" + "=" * 80)
    print("✓ Cross-document relationships built successfully!")
    print("=" * 80)


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
    parser.add_argument(
        "--no-relationships",
        action="store_true",
        help="Skip cross-document relationship building after ingestion (auto-ingest mode only)"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.85,
        help="Minimum similarity for semantic links (default: 0.85)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of similar chunks to find per source chunk (default: 10)"
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        choices=["gemini", "mistral"],
        default="gemini",
        help="LLM provider for AAS classification (default: gemini)"
    )
    parser.add_argument(
        "--classify-aas",
        action="store_true",
        help="Classify PDFs to AAS (Asset Administration Shell) submodels using LLM"
    )
    parser.add_argument(
        "--extract-aas",
        action="store_true",
        help="Extract structured data for AAS submodels (requires --classify-aas first)"
    )
    parser.add_argument(
        "--validate-aas",
        action="store_true",
        help="Validate and complete AAS extracted data (requires --extract-aas first)"
    )
    parser.add_argument(
        "--generate-aas-xml",
        action="store_true",
        help="Generate AAS v5.0 XML from validated data (requires --validate-aas first)"
    )

    args = parser.parse_args()

    # Initialize storage backend
    storage = None if args.no_db else get_storage_backend()

    # Setup output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Special mode: AAS XML Generation only (no ingestion)
    if args.generate_aas_xml and not args.pdf and not args.extract_aas and not args.classify_aas and not args.validate_aas:
        print("=" * 80)
        print("AAS XML GENERATION MODE")
        print("=" * 80)
        print(f"🤖 LLM Provider: {args.llm_provider}")
        print()

        if not storage:
            print("Error: --generate-aas-xml requires database storage (remove --no-db)", file=sys.stderr)
            sys.exit(1)

        try:
            xml_output = generate_aas_xml(
                storage=storage,
                llm_provider=args.llm_provider,
                output_path=None  # Use default path
            )
            print("\n✅ XML generation complete!")
            sys.exit(0)

        except Exception as e:
            print(f"\n❌ Error during XML generation: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Special mode: AAS Validation only (no ingestion)
    if args.validate_aas and not args.pdf and not args.extract_aas and not args.classify_aas:
        print("=" * 80)
        print("AAS VALIDATION MODE")
        print("=" * 80)
        print(f"🤖 LLM Provider: {args.llm_provider}")
        print()

        if not storage:
            print("Error: --validate-aas requires database storage (remove --no-db)", file=sys.stderr)
            sys.exit(1)

        try:
            completed_data, validation_report = validate_aas_data(
                storage=storage,
                llm_provider=args.llm_provider
            )
            print("\n✅ Validation complete!")
            print(f"   Status: {'✅ Complete' if validation_report.get('is_complete') else '⚠️  Incomplete'}")
            sys.exit(0)

        except Exception as e:
            print(f"\n❌ Error during validation: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Special mode: AAS Extraction only (no ingestion)
    if args.extract_aas and not args.pdf and not args.classify_aas and not args.validate_aas:
        print("=" * 80)
        print("AAS EXTRACTION MODE")
        print("=" * 80)
        print(f"🤖 LLM Provider: {args.llm_provider}")
        print()

        if not storage:
            print("Error: --extract-aas requires database storage (remove --no-db)", file=sys.stderr)
            sys.exit(1)

        try:
            extract_aas_data(
                storage=storage,
                llm_provider=args.llm_provider
            )
            print("\n✅ Extraction complete!")
            sys.exit(0)

        except Exception as e:
            print(f"\n❌ Error during extraction: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Special mode: AAS Classification only (no ingestion)
    if args.classify_aas and not args.pdf and not args.extract_aas:
        print("=" * 80)
        print("AAS CLASSIFICATION MODE")
        print("=" * 80)
        print(f"🤖 LLM Provider: {args.llm_provider}")
        print()

        if not storage:
            print("Error: --classify-aas requires database storage (remove --no-db)", file=sys.stderr)
            sys.exit(1)

        try:
            classify_pdfs_to_aas(
                storage=storage,
                llm_provider=args.llm_provider
            )
            print("\n✅ Classification complete!")
            sys.exit(0)

        except Exception as e:
            print(f"\n❌ Error during classification: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Special mode: AAS Full Pipeline (no ingestion)
    if args.classify_aas and args.extract_aas and not args.pdf:
        if args.generate_aas_xml:
            print("=" * 80)
            print("AAS FULL PIPELINE: CLASSIFY → EXTRACT → VALIDATE → GENERATE XML")
            print("=" * 80)
        elif args.validate_aas:
            print("=" * 80)
            print("AAS FULL PIPELINE: CLASSIFY → EXTRACT → VALIDATE")
            print("=" * 80)
        else:
            print("=" * 80)
            print("AAS CLASSIFICATION + EXTRACTION MODE")
            print("=" * 80)

        print(f"🤖 LLM Provider: {args.llm_provider}")
        print()

        if not storage:
            print("Error: AAS operations require database storage (remove --no-db)", file=sys.stderr)
            sys.exit(1)

        try:
            # Phase 1: Classify
            classify_pdfs_to_aas(
                storage=storage,
                llm_provider=args.llm_provider
            )

            # Phase 2: Extract
            extract_aas_data(
                storage=storage,
                llm_provider=args.llm_provider
            )

            # Phase 3: Validate (if requested)
            if args.validate_aas:
                completed_data, validation_report = validate_aas_data(
                    storage=storage,
                    llm_provider=args.llm_provider
                )

                # Phase 4: Generate XML (if requested)
                if args.generate_aas_xml:
                    xml_output = generate_aas_xml(
                        storage=storage,
                        llm_provider=args.llm_provider,
                        output_path=None  # Use default path
                    )
                    print("\n✅ Full AAS pipeline complete (all 4 phases)!")
                    print(f"   Validation Status: {'✅ Complete' if validation_report.get('is_complete') else '⚠️  Incomplete'}")
                else:
                    print("\n✅ AAS pipeline complete (phases 1-3)!")
                    print(f"   Status: {'✅ Complete' if validation_report.get('is_complete') else '⚠️  Incomplete'}")
            else:
                print("\n✅ Classification and extraction complete!")

            sys.exit(0)

        except Exception as e:
            print(f"\n❌ Error during AAS processing: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)

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

                # Build cross-document relationships
                if storage and not args.no_db and not args.no_relationships:
                    build_cross_document_relationships(
                        storage=storage,
                        similarity_threshold=args.similarity_threshold,
                        top_k=args.top_k
                    )

                # Classify PDFs to AAS submodels
                if storage and not args.no_db and args.classify_aas:
                    classify_pdfs_to_aas(
                        storage=storage,
                        llm_provider=args.llm_provider
                    )

                    # Extract AAS data if requested
                    if args.extract_aas:
                        extract_aas_data(
                            storage=storage,
                            llm_provider=args.llm_provider
                        )

                        # Validate AAS data if requested
                        if args.validate_aas:
                            completed_data, validation_report = validate_aas_data(
                                storage=storage,
                                llm_provider=args.llm_provider
                            )

                            # Generate AAS XML if requested
                            if args.generate_aas_xml:
                                xml_output = generate_aas_xml(
                                    storage=storage,
                                    llm_provider=args.llm_provider,
                                    output_path=None  # Use default path
                                )

        except Exception as e:
            print(f"\n❌ Error during auto-ingestion: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
