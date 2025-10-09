#!/usr/bin/env python3
"""
Build cross-document knowledge graph by analyzing relationships between PDFs.

This script implements a 3-phase analysis pipeline:

Phase 1 (MVP):
  - Cross-document references (explicit)
  - Named entity extraction and linking
  - Document versioning detection

Phase 2 (Expansion):
  - Semantic similarity between chunks
  - Topic clustering

Phase 3 (Advanced):
  - Citation network analysis

Usage:
    # Run all phases
    python build_cross_document_graph.py

    # Run specific phases
    python build_cross_document_graph.py --phase 1
    python build_cross_document_graph.py --phase 1 2

    # With specific parameters
    python build_cross_document_graph.py --similarity-threshold 0.9 --top-k 5
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

# Fix for macOS multiprocessing
import multiprocessing
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

# Fix FAISS threading issues
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from pdfkg.storage import get_storage_backend
from pdfkg.cross_document import CrossDocumentAnalyzer, cross_doc_ref_to_dict, semantic_link_to_dict, version_relation_to_dict
from pdfkg.ner import TechnicalNER, extract_entities_from_chunks, entity_to_dict


def run_phase_1(analyzer: CrossDocumentAnalyzer, storage, all_pdfs: list):
    """
    Phase 1: Cross-doc refs, Named entities, Document versioning.

    Low computational cost, high value.
    """
    print("\n" + "=" * 80)
    print("PHASE 1: MVP - Cross-doc refs, Entities, Versioning")
    print("=" * 80)

    # 1.1: Cross-document references
    print("\n📎 Step 1.1: Extracting cross-document references...")
    all_refs = []
    for pdf in all_pdfs:
        slug = pdf['slug']
        print(f"  Processing {pdf['filename']}...")

        refs = analyzer.extract_cross_doc_refs(slug)
        resolved_refs = analyzer.resolve_cross_doc_refs(refs)
        all_refs.extend(resolved_refs)

        print(f"    Found {len(refs)} references ({sum(1 for r in resolved_refs if r.target_pdf)} resolved)")

    # Save cross-doc refs to ArangoDB
    if all_refs:
        print(f"\n  Saving {len(all_refs)} cross-document references...")
        # TODO: Add save_cross_doc_refs method to ArangoDBClient
        # For now, save to metadata
        storage.db_client.save_metadata('__global__', 'cross_doc_refs', [cross_doc_ref_to_dict(r) for r in all_refs])

    # 1.2: Named entity extraction
    print("\n🏷️  Step 1.2: Extracting named entities...")
    all_entities = {}
    ner = TechnicalNER()

    for pdf in all_pdfs:
        slug = pdf['slug']
        print(f"  Processing {pdf['filename']}...")

        chunks = storage.get_chunks(slug)
        entity_dict = extract_entities_from_chunks(chunks, include_products=True)

        total_entities = sum(len(entities) for entities in entity_dict.values())
        all_entities[slug] = entity_dict
        print(f"    Found {total_entities} entities in {len(chunks)} chunks")

    # Save entities to metadata
    print(f"\n  Saving entities...")
    for slug, entity_dict in all_entities.items():
        # Convert to serializable format
        serializable = {}
        for chunk_id, entities in entity_dict.items():
            serializable[chunk_id] = [entity_to_dict(e) for e in entities]
        storage.db_client.save_metadata(slug, 'extracted_entities', serializable)

    # 1.3: Document versioning
    print("\n📅 Step 1.3: Detecting version relationships...")
    version_rels = analyzer.detect_version_relationships(all_pdfs)
    print(f"  Found {len(version_rels)} version relationships")

    for rel in version_rels:
        pdf_a_name = next(p['filename'] for p in all_pdfs if p['slug'] == rel.pdf_from)
        pdf_b_name = next(p['filename'] for p in all_pdfs if p['slug'] == rel.pdf_to)
        print(f"    {pdf_a_name} ↔ {pdf_b_name} (similarity: {rel.similarity:.2f})")

    # Save version relationships
    if version_rels:
        print(f"\n  Saving version relationships...")
        storage.db_client.save_metadata('__global__', 'version_relations', [version_relation_to_dict(r) for r in version_rels])

    print("\n✅ Phase 1 complete!")
    return {
        'cross_doc_refs': all_refs,
        'entities': all_entities,
        'version_relations': version_rels
    }


def run_phase_2(analyzer: CrossDocumentAnalyzer, storage, all_pdfs: list, similarity_threshold: float, top_k: int):
    """
    Phase 2: Semantic similarity, Topic clustering.

    Higher computational cost, provides global overview.
    """
    print("\n" + "=" * 80)
    print("PHASE 2: EXPANSION - Semantic Similarity & Topic Clustering")
    print("=" * 80)

    # 2.1: Semantic similarity
    print(f"\n🔍 Step 2.1: Finding semantic similarities (threshold={similarity_threshold}, top_k={top_k})...")
    print("  ⚠️  This may take several minutes for large document collections...")

    all_semantic_links = []
    for pdf in all_pdfs:
        slug = pdf['slug']
        print(f"\n  Analyzing {pdf['filename']}...")

        links = analyzer.find_semantic_similarities(slug, threshold=similarity_threshold, top_k=top_k)
        all_semantic_links.extend(links)
        print(f"    Found {len(links)} semantic links")

    # Save semantic links
    if all_semantic_links:
        print(f"\n  Saving {len(all_semantic_links)} semantic links...")
        storage.db_client.save_metadata('__global__', 'semantic_links', [semantic_link_to_dict(l) for l in all_semantic_links])

    # 2.2: Topic clustering
    print("\n📊 Step 2.2: Clustering documents by topics...")
    n_topics = min(10, max(2, len(all_pdfs) // 2))  # Adaptive number of topics
    topics_result = analyzer.cluster_documents_by_topic(all_pdfs, n_topics=n_topics)

    if topics_result:
        print(f"  Found {topics_result['n_topics']} topics")
        print(f"  Silhouette score: {topics_result['silhouette_score']:.3f}")

        # Print topic summary
        for topic_key, topic_data in topics_result['topics'].items():
            print(f"\n  {topic_key}: {topic_data['num_documents']} documents")
            for doc_slug in topic_data['documents'][:3]:  # Show first 3
                doc_name = next((p['filename'] for p in all_pdfs if p['slug'] == doc_slug), doc_slug)
                print(f"    - {doc_name}")

        # Save topics
        print(f"\n  Saving topic clustering results...")
        storage.db_client.save_metadata('__global__', 'topics', topics_result)

    print("\n✅ Phase 2 complete!")
    return {
        'semantic_links': all_semantic_links,
        'topics': topics_result
    }


def run_phase_3(analyzer: CrossDocumentAnalyzer, storage, all_pdfs: list):
    """
    Phase 3: Citation network.

    Advanced analysis for understanding document influence and relationships.
    """
    print("\n" + "=" * 80)
    print("PHASE 3: ADVANCED - Citation Network")
    print("=" * 80)

    print("\n📚 Building citation network...")
    citations = analyzer.build_citation_network(all_pdfs)

    print(f"  Found {len(citations)} citation relationships")

    # Analyze citation patterns
    if citations:
        # Most cited documents
        citation_counts = {}
        for cite in citations:
            target = cite['target']
            citation_counts[target] = citation_counts.get(target, 0) + 1

        sorted_citations = sorted(citation_counts.items(), key=lambda x: x[1], reverse=True)

        print("\n  Most cited documents:")
        for slug, count in sorted_citations[:5]:
            doc_name = next((p['filename'] for p in all_pdfs if p['slug'] == slug), slug)
            print(f"    {doc_name}: {count} citations")

        # Save citation network
        print(f"\n  Saving citation network...")
        storage.db_client.save_metadata('__global__', 'citation_network', citations)

    print("\n✅ Phase 3 complete!")
    return {'citations': citations}


def main():
    parser = argparse.ArgumentParser(
        description="Build cross-document knowledge graph from PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--phase',
        nargs='+',
        type=int,
        choices=[1, 2, 3],
        default=[1, 2, 3],
        help='Phases to run (default: all phases)'
    )
    parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.85,
        help='Minimum similarity for semantic links (default: 0.85)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of similar chunks to find per source chunk (default: 10)'
    )
    parser.add_argument(
        '--no-milvus',
        action='store_true',
        help='Skip phases that require Milvus (semantic similarity, topic clustering)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CROSS-DOCUMENT KNOWLEDGE GRAPH BUILDER")
    print("=" * 80)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Phases to run: {sorted(args.phase)}")

    # Initialize storage
    print("\n🔌 Connecting to storage backend...")
    storage = get_storage_backend()

    # Get Milvus client if available
    milvus_client = None
    if not args.no_milvus and hasattr(storage, 'milvus_client') and storage.milvus_client:
        milvus_client = storage.milvus_client
        print("✅ Milvus client available for semantic search")
    else:
        print("⚠️  Milvus not available - will skip semantic similarity and topic clustering")
        # Remove phase 2 if Milvus not available
        if 2 in args.phase:
            print("   Removing Phase 2 from execution plan")
            args.phase = [p for p in args.phase if p != 2]

    # Get all PDFs
    print("\n📚 Loading PDFs from database...")
    all_pdfs = storage.list_pdfs()

    if len(all_pdfs) < 2:
        print("❌ Error: Need at least 2 PDFs for cross-document analysis")
        print(f"   Found {len(all_pdfs)} PDF(s)")
        print("   Please process more PDFs first using: python cli.py")
        sys.exit(1)

    print(f"✅ Found {len(all_pdfs)} PDFs:")
    for pdf in all_pdfs:
        print(f"   - {pdf['filename']} ({pdf['num_chunks']} chunks)")

    # Initialize analyzer
    analyzer = CrossDocumentAnalyzer(storage, milvus_client)

    # Run phases
    results = {}

    if 1 in args.phase:
        results['phase1'] = run_phase_1(analyzer, storage, all_pdfs)

    if 2 in args.phase:
        results['phase2'] = run_phase_2(analyzer, storage, all_pdfs, args.similarity_threshold, args.top_k)

    if 3 in args.phase:
        results['phase3'] = run_phase_3(analyzer, storage, all_pdfs)

    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"End time: {datetime.now().isoformat()}")

    if 1 in args.phase:
        print(f"\n📊 Phase 1 Results:")
        print(f"   - Cross-doc refs: {len(results.get('phase1', {}).get('cross_doc_refs', []))}")
        print(f"   - Version relations: {len(results.get('phase1', {}).get('version_relations', []))}")

    if 2 in args.phase:
        print(f"\n📊 Phase 2 Results:")
        print(f"   - Semantic links: {len(results.get('phase2', {}).get('semantic_links', []))}")
        topics_data = results.get('phase2', {}).get('topics', {})
        if topics_data:
            print(f"   - Topics: {topics_data.get('n_topics', 0)}")

    if 3 in args.phase:
        print(f"\n📊 Phase 3 Results:")
        print(f"   - Citations: {len(results.get('phase3', {}).get('citations', []))}")

    print("\n💡 Next steps:")
    print("   - Explore relationships: python explore_cross_document.py")
    print("   - Query with cross-doc context: python chatbot.py")
    print("   - View in ArangoDB UI: http://localhost:8529")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
