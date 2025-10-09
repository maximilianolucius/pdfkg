#!/usr/bin/env python3
"""
Explore cross-document relationships discovered in the knowledge graph.

This script provides interactive queries to explore:
- Cross-document references
- Shared entities across PDFs
- Semantic similarities between chunks
- Version relationships
- Topic clusters
- Citation networks

Usage:
    python explore_cross_document.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from pdfkg.storage import get_storage_backend
from datetime import datetime


def explore_cross_doc_refs(storage):
    """Explore cross-document references."""
    print("\n" + "=" * 80)
    print("CROSS-DOCUMENT REFERENCES")
    print("=" * 80)

    try:
        refs_data = storage.db_client.get_metadata('__global__', 'cross_doc_refs') or []
        print(f"\nTotal cross-document references: {len(refs_data)}")

        if not refs_data:
            print("  No cross-document references found. Run build_cross_document_graph.py first.")
            return

        # Group by source PDF
        refs_by_source = {}
        for ref in refs_data:
            source = ref['source_pdf']
            refs_by_source.setdefault(source, []).append(ref)

        # Display
        all_pdfs = storage.list_pdfs()
        pdf_names = {p['slug']: p['filename'] for p in all_pdfs}

        for source_slug, refs in sorted(refs_by_source.items(), key=lambda x: len(x[1]), reverse=True):
            source_name = pdf_names.get(source_slug, source_slug)
            print(f"\n📄 {source_name}")
            print(f"   {len(refs)} references:")

            for ref in refs[:5]:  # Show first 5
                target_name = pdf_names.get(ref['target_pdf'], ref['target_document_name'])
                print(f"   → {target_name}")
                print(f"      \"{ref['mention_text']}\"")
                if ref.get('target_section'):
                    print(f"      Section: {ref['target_section']}")

            if len(refs) > 5:
                print(f"   ... and {len(refs) - 5} more")

    except Exception as e:
        print(f"Error: {e}")


def explore_entities(storage):
    """Explore shared entities across documents."""
    print("\n" + "=" * 80)
    print("SHARED ENTITIES ACROSS DOCUMENTS")
    print("=" * 80)

    all_pdfs = storage.list_pdfs()

    # Collect all entities
    entity_occurrences = {}  # {entity_text: [(pdf_slug, count), ...]}

    for pdf in all_pdfs:
        try:
            entities_data = storage.db_client.get_metadata(pdf['slug'], 'extracted_entities') or {}

            # Count entities in this PDF
            entity_counts = {}
            for chunk_id, entities_list in entities_data.items():
                for entity_dict in entities_list:
                    key = (entity_dict['text'].lower(), entity_dict['type'])
                    entity_counts[key] = entity_counts.get(key, 0) + 1

            # Add to global occurrences
            for (text, entity_type), count in entity_counts.items():
                if (text, entity_type) not in entity_occurrences:
                    entity_occurrences[(text, entity_type)] = []
                entity_occurrences[(text, entity_type)].append((pdf['slug'], pdf['filename'], count))

        except Exception:
            continue

    # Find shared entities (appear in multiple PDFs)
    shared = {k: v for k, v in entity_occurrences.items() if len(v) > 1}

    print(f"\nTotal unique entities: {len(entity_occurrences)}")
    print(f"Shared across PDFs: {len(shared)}")

    if shared:
        # Sort by number of PDFs
        sorted_shared = sorted(shared.items(), key=lambda x: len(x[1]), reverse=True)

        print("\n🏷️  Top shared entities:")
        for (text, entity_type), occurrences in sorted_shared[:15]:
            print(f"\n  \"{text}\" ({entity_type})")
            print(f"    Appears in {len(occurrences)} documents:")
            for slug, filename, count in occurrences:
                print(f"      - {filename} ({count}x)")
    else:
        print("\n  No shared entities found. Run build_cross_document_graph.py first.")


def explore_semantic_links(storage):
    """Explore semantic similarity links."""
    print("\n" + "=" * 80)
    print("SEMANTIC SIMILARITY LINKS")
    print("=" * 80)

    try:
        links_data = storage.db_client.get_metadata('__global__', 'semantic_links') or []
        print(f"\nTotal semantic links: {len(links_data)}")

        if not links_data:
            print("  No semantic links found. Run build_cross_document_graph.py --phase 2")
            return

        # Group by PDF pair
        pdf_pairs = {}
        for link in links_data:
            pair = tuple(sorted([link['source_pdf'], link['target_pdf']]))
            pdf_pairs.setdefault(pair, []).append(link)

        all_pdfs = storage.list_pdfs()
        pdf_names = {p['slug']: p['filename'] for p in all_pdfs}

        # Sort by number of links
        sorted_pairs = sorted(pdf_pairs.items(), key=lambda x: len(x[1]), reverse=True)

        print("\n🔗 Document pairs with semantic links:")
        for (pdf_a, pdf_b), pair_links in sorted_pairs[:10]:
            name_a = pdf_names.get(pdf_a, pdf_a)
            name_b = pdf_names.get(pdf_b, pdf_b)
            avg_sim = sum(l['similarity'] for l in pair_links) / len(pair_links)

            print(f"\n  {name_a} ↔ {name_b}")
            print(f"    {len(pair_links)} links, avg similarity: {avg_sim:.3f}")

            # Show top link
            top_link = max(pair_links, key=lambda l: l['similarity'])
            print(f"    Strongest link: {top_link['similarity']:.3f}")

    except Exception as e:
        print(f"Error: {e}")


def explore_version_relations(storage):
    """Explore version relationships."""
    print("\n" + "=" * 80)
    print("VERSION RELATIONSHIPS")
    print("=" * 80)

    try:
        relations = storage.db_client.get_metadata('__global__', 'version_relations') or []
        print(f"\nTotal version relationships: {len(relations)}")

        if not relations:
            print("  No version relationships found. Run build_cross_document_graph.py first.")
            return

        all_pdfs = storage.list_pdfs()
        pdf_names = {p['slug']: p['filename'] for p in all_pdfs}

        print("\n📅 Document versions:")
        for rel in relations:
            name_from = pdf_names.get(rel['pdf_from'], rel['pdf_from'])
            name_to = pdf_names.get(rel['pdf_to'], rel['pdf_to'])

            print(f"\n  {name_from}")
            print(f"  ↓ ({rel['relationship_type']}, similarity: {rel['similarity']:.2f})")
            print(f"  {name_to}")

            if rel.get('version_from') and rel.get('version_to'):
                print(f"    Versions: {rel['version_from']} → {rel['version_to']}")

    except Exception as e:
        print(f"Error: {e}")


def explore_topics(storage):
    """Explore topic clusters."""
    print("\n" + "=" * 80)
    print("TOPIC CLUSTERS")
    print("=" * 80)

    try:
        topics_data = storage.db_client.get_metadata('__global__', 'topics') or {}
        topics = topics_data.get('topics', {})

        if not topics:
            print("  No topics found. Run build_cross_document_graph.py --phase 2")
            return

        print(f"\nTotal topics: {len(topics)}")
        print(f"Silhouette score: {topics_data.get('silhouette_score', 0):.3f}")

        all_pdfs = storage.list_pdfs()
        pdf_names = {p['slug']: p['filename'] for p in all_pdfs}

        print("\n📊 Topics:")
        for topic_key, topic_data in topics.items():
            print(f"\n  {topic_key} ({topic_data['num_documents']} documents)")
            for doc_slug in topic_data['documents']:
                doc_name = pdf_names.get(doc_slug, doc_slug)
                print(f"    - {doc_name}")

    except Exception as e:
        print(f"Error: {e}")


def explore_citation_network(storage):
    """Explore citation network."""
    print("\n" + "=" * 80)
    print("CITATION NETWORK")
    print("=" * 80)

    try:
        citations = storage.db_client.get_metadata('__global__', 'citation_network') or []
        print(f"\nTotal citations: {len(citations)}")

        if not citations:
            print("  No citations found. Run build_cross_document_graph.py --phase 3")
            return

        all_pdfs = storage.list_pdfs()
        pdf_names = {p['slug']: p['filename'] for p in all_pdfs}

        # Calculate citation counts
        incoming = {}  # How many times cited
        outgoing = {}  # How many citations made

        for cite in citations:
            incoming[cite['target']] = incoming.get(cite['target'], 0) + cite['weight']
            outgoing[cite['source']] = outgoing.get(cite['source'], 0) + cite['weight']

        # Most cited
        print("\n📚 Most cited documents:")
        for slug, count in sorted(incoming.items(), key=lambda x: x[1], reverse=True)[:5]:
            name = pdf_names.get(slug, slug)
            print(f"  {name}: {count} citations")

        # Most citing
        print("\n📝 Documents with most citations:")
        for slug, count in sorted(outgoing.items(), key=lambda x: x[1], reverse=True)[:5]:
            name = pdf_names.get(slug, slug)
            print(f"  {name}: {count} outgoing citations")

    except Exception as e:
        print(f"Error: {e}")


def main():
    print("=" * 80)
    print("CROSS-DOCUMENT KNOWLEDGE GRAPH EXPLORER")
    print("=" * 80)

    # Connect to storage
    print("\n🔌 Connecting to database...")
    storage = get_storage_backend()

    # Check for PDFs
    all_pdfs = storage.list_pdfs()
    if len(all_pdfs) < 2:
        print(f"\n❌ Need at least 2 PDFs. Found: {len(all_pdfs)}")
        print("   Process PDFs first: python cli.py")
        return

    print(f"✅ Found {len(all_pdfs)} PDFs")

    # Run explorations
    explore_cross_doc_refs(storage)
    explore_entities(storage)
    explore_version_relations(storage)
    explore_semantic_links(storage)
    explore_topics(storage)
    explore_citation_network(storage)

    print("\n" + "=" * 80)
    print("EXPLORATION COMPLETE")
    print("=" * 80)
    print("\n💡 To rebuild relationships, run:")
    print("   python build_cross_document_graph.py")


if __name__ == "__main__":
    main()
