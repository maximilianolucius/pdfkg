"""
Cross-document relationship analysis for PDF knowledge graphs.

Implements three phases:
- Phase 1 (MVP): Cross-doc refs, Named entities, Document versioning
- Phase 2 (Expansion): Semantic similarity, Topic clustering
- Phase 3 (Advanced): Citation network
"""

import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from fuzzywuzzy import fuzz


@dataclass
class CrossDocRef:
    """Cross-document reference."""
    source_pdf: str
    source_chunk: str
    mention_text: str
    target_document_name: str
    target_pdf: Optional[str] = None
    target_section: Optional[str] = None
    target_page: Optional[int] = None
    confidence: float = 1.0
    resolution_method: str = "unknown"


@dataclass
class SemanticLink:
    """Semantic similarity link between chunks."""
    source_pdf: str
    source_chunk: str
    target_pdf: str
    target_chunk: str
    similarity: float
    method: str = "milvus_cosine"


@dataclass
class VersionRelation:
    """Version relationship between documents."""
    pdf_from: str
    pdf_to: str
    version_from: Optional[str] = None
    version_to: Optional[str] = None
    similarity: float = 0.0
    relationship_type: str = "version"


# =============================================================================
# PHASE 1: CROSS-DOCUMENT REFERENCES
# =============================================================================

CROSS_DOC_PATTERNS = [
    # "see Installation Manual section 4"
    r'(?:see|refer\s+to|consult|check)\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(Manual|Guide|Datasheet|Document)(?:\s+section\s+(\S+)|page\s+(\d+))?',

    # "refer to Troubleshooting Guide, page 42"
    r'(?:see|refer\s+to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(Manual|Guide),?\s+(?:section\s+(\S+)|page\s+(\d+))',

    # Document codes: "DOC-12345-EN"
    r'\b([A-Z]{2,4}-\d{3,6}(?:-[A-Z0-9]+)?)\b',

    # "as described in Installation Manual"
    r'(?:as\s+)?(?:described|explained|detailed|shown)\s+in\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(Manual|Guide|Datasheet|Document)',
]


class CrossDocumentAnalyzer:
    """Analyzes relationships between documents."""

    def __init__(self, storage, milvus_client=None):
        """
        Initialize analyzer.

        Args:
            storage: Storage backend (ArangoDB)
            milvus_client: Optional Milvus client for semantic search
        """
        self.storage = storage
        self.milvus_client = milvus_client

    # =========================================================================
    # PHASE 1.1: CROSS-DOCUMENT REFERENCES
    # =========================================================================

    def extract_cross_doc_refs(self, pdf_slug: str) -> List[CrossDocRef]:
        """
        Extract cross-document references from a PDF.

        Args:
            pdf_slug: PDF identifier

        Returns:
            List of CrossDocRef objects
        """
        refs = []
        chunks = self.storage.get_chunks(pdf_slug)

        for chunk in chunks:
            chunk_id = chunk.get('chunk_id') or chunk.get('id')
            text = chunk.get('text', '')

            # Apply patterns
            for pattern in CROSS_DOC_PATTERNS:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    mention_text = match.group(0)

                    # Extract document name (group 1 for most patterns)
                    doc_name = match.group(1) if match.lastindex >= 1 else mention_text

                    # Extract section/page if present
                    section = None
                    page = None

                    if match.lastindex >= 3 and match.group(3):
                        section = match.group(3)
                    if match.lastindex >= 4 and match.group(4):
                        page = int(match.group(4))

                    refs.append(CrossDocRef(
                        source_pdf=pdf_slug,
                        source_chunk=chunk_id,
                        mention_text=mention_text,
                        target_document_name=doc_name,
                        target_section=section,
                        target_page=page,
                        confidence=0.8,  # Pattern-based, medium-high confidence
                    ))

        return refs

    def resolve_cross_doc_refs(self, refs: List[CrossDocRef]) -> List[CrossDocRef]:
        """
        Resolve cross-document references to actual PDF slugs.

        Args:
            refs: List of unresolved references

        Returns:
            List of references with target_pdf resolved where possible
        """
        all_pdfs = self.storage.list_pdfs()

        for ref in refs:
            target_slug = self._resolve_document_name(ref.target_document_name, all_pdfs)
            if target_slug:
                ref.target_pdf = target_slug
                ref.resolution_method = "fuzzy_match"

        return refs

    def _resolve_document_name(self, doc_name: str, all_pdfs: List[Dict]) -> Optional[str]:
        """
        Resolve document name to PDF slug.

        Args:
            doc_name: Document name from reference
            all_pdfs: List of all PDF metadata

        Returns:
            PDF slug if found, None otherwise
        """
        # 1. Exact match in filename
        for pdf in all_pdfs:
            if doc_name.lower() in pdf['filename'].lower():
                return pdf['slug']

        # 2. Fuzzy match
        from fuzzywuzzy import process
        filenames = {p['filename']: p['slug'] for p in all_pdfs}

        if filenames:
            best_match, score = process.extractOne(doc_name, filenames.keys())
            if score > 80:
                return filenames[best_match]

        # 3. Check metadata title
        for pdf in all_pdfs:
            metadata = pdf.get('metadata', {})
            if isinstance(metadata, dict):
                title = metadata.get('title', '')
                if title and doc_name.lower() in title.lower():
                    return pdf['slug']

        return None

    # =========================================================================
    # PHASE 1.2: DOCUMENT VERSIONING
    # =========================================================================

    def detect_version_relationships(self, all_pdfs: List[Dict]) -> List[VersionRelation]:
        """
        Detect version relationships between PDFs.

        Args:
            all_pdfs: List of all PDF metadata

        Returns:
            List of VersionRelation objects
        """
        relations = []

        # Compare each pair of PDFs
        for i, pdf_a in enumerate(all_pdfs):
            for pdf_b in all_pdfs[i+1:]:
                relation = self._check_version_relationship(pdf_a, pdf_b)
                if relation:
                    relations.append(relation)

        return relations

    def _check_version_relationship(self, pdf_a: Dict, pdf_b: Dict) -> Optional[VersionRelation]:
        """
        Check if two PDFs are versions of the same document.

        Args:
            pdf_a: First PDF metadata
            pdf_b: Second PDF metadata

        Returns:
            VersionRelation if they are versions, None otherwise
        """
        # 1. Filename similarity (without version numbers)
        name_a = self._strip_version_from_filename(pdf_a['filename'])
        name_b = self._strip_version_from_filename(pdf_b['filename'])

        filename_sim = fuzz.ratio(name_a, name_b)

        if filename_sim < 85:
            return None  # Not similar enough

        # 2. Extract versions
        version_a = self._extract_version(pdf_a['filename'])
        version_b = self._extract_version(pdf_b['filename'])

        # 3. ToC structure similarity
        toc_sim = self._compare_toc_structure(pdf_a['slug'], pdf_b['slug'])

        # 4. Content overlap (if available)
        # For now, use simple heuristic based on page count
        page_ratio = min(pdf_a['num_pages'], pdf_b['num_pages']) / max(pdf_a['num_pages'], pdf_b['num_pages'])

        # Decision: version relationship if filename similar and reasonable page ratio
        if filename_sim > 85 and page_ratio > 0.7:
            return VersionRelation(
                pdf_from=pdf_a['slug'],
                pdf_to=pdf_b['slug'],
                version_from=version_a,
                version_to=version_b,
                similarity=filename_sim / 100.0,
                relationship_type='version_of'
            )

        return None

    def _strip_version_from_filename(self, filename: str) -> str:
        """Remove version indicators from filename."""
        # Remove common version patterns
        name = re.sub(r'[_\-\s]?v?\d+\.\d+(?:\.\d+)?', '', filename, flags=re.IGNORECASE)
        name = re.sub(r'[_\-\s]?rev\d+', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\.pdf$', '', name, flags=re.IGNORECASE)
        return name

    def _extract_version(self, filename: str) -> Optional[str]:
        """Extract version number from filename."""
        # Pattern: v1.2.3, v1.2, 1.2.3, etc.
        patterns = [
            r'v?(\d+\.\d+(?:\.\d+)?)',
            r'rev(\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _compare_toc_structure(self, slug_a: str, slug_b: str) -> float:
        """
        Compare ToC structure similarity.

        Args:
            slug_a: First PDF slug
            slug_b: Second PDF slug

        Returns:
            Similarity score 0-1
        """
        try:
            toc_a = self.storage.get_metadata(slug_a, 'toc') or []
            toc_b = self.storage.get_metadata(slug_b, 'toc') or []

            if not toc_a or not toc_b:
                return 0.5  # Unknown

            # Compare section titles
            titles_a = [entry.get('title', '') for entry in toc_a]
            titles_b = [entry.get('title', '') for entry in toc_b]

            # Use set intersection
            set_a = set(titles_a)
            set_b = set(titles_b)

            if not set_a or not set_b:
                return 0.5

            intersection = len(set_a & set_b)
            union = len(set_a | set_b)

            return intersection / union if union > 0 else 0.0

        except Exception:
            return 0.5  # Default to unknown

    # =========================================================================
    # PHASE 2.1: SEMANTIC SIMILARITY
    # =========================================================================

    def find_semantic_similarities(
        self,
        pdf_slug: str,
        threshold: float = 0.85,
        top_k: int = 10
    ) -> List[SemanticLink]:
        """
        Find semantically similar chunks across documents.

        Args:
            pdf_slug: Source PDF slug
            threshold: Minimum similarity threshold (0-1)
            top_k: Number of similar chunks to retrieve per source chunk

        Returns:
            List of SemanticLink objects
        """
        if not self.milvus_client:
            print("⚠️  Milvus client not available, skipping semantic similarity")
            return []

        links = []
        all_pdfs = self.storage.list_pdfs()

        # Get source PDF chunks and embeddings
        try:
            embeddings, chunk_ids = self.milvus_client.load_embeddings(pdf_slug)
        except Exception as e:
            print(f"⚠️  Failed to load embeddings for {pdf_slug}: {e}")
            return []

        # For each chunk, search in other PDFs
        for idx, (embedding, source_chunk_id) in enumerate(zip(embeddings, chunk_ids)):
            if idx % 100 == 0:
                print(f"  Processing chunk {idx}/{len(embeddings)}...")

            # Search in each other PDF
            for target_pdf in all_pdfs:
                target_slug = target_pdf['slug']

                if target_slug == pdf_slug:
                    continue  # Skip self

                try:
                    # Search in target PDF
                    distances, indices = self.milvus_client.search(
                        pdf_slug=target_slug,
                        query_embedding=embedding,
                        top_k=min(top_k, 5)  # Limit to avoid explosion
                    )

                    # Get target chunk IDs
                    target_embeddings, target_chunk_ids = self.milvus_client.load_embeddings(target_slug)

                    # Filter by threshold
                    for dist, idx_target in zip(distances, indices):
                        if dist >= threshold:
                            links.append(SemanticLink(
                                source_pdf=pdf_slug,
                                source_chunk=source_chunk_id,
                                target_pdf=target_slug,
                                target_chunk=target_chunk_ids[idx_target],
                                similarity=float(dist),
                                method="milvus_cosine"
                            ))

                except Exception as e:
                    print(f"  Warning: Failed to search in {target_slug}: {e}")
                    continue

        return links

    # =========================================================================
    # PHASE 2.2: TOPIC CLUSTERING
    # =========================================================================

    def cluster_documents_by_topic(
        self,
        all_pdfs: List[Dict],
        n_topics: int = 10
    ) -> Dict[str, Any]:
        """
        Cluster documents by topics using embeddings.

        Args:
            all_pdfs: List of all PDF metadata
            n_topics: Number of topics to extract

        Returns:
            Dictionary with topic assignments and metadata
        """
        if not self.milvus_client:
            print("⚠️  Milvus client not available, skipping topic clustering")
            return {}

        print(f"📊 Clustering {len(all_pdfs)} documents into {n_topics} topics...")

        # 1. Compute document-level embeddings (average pooling)
        doc_embeddings = []
        doc_slugs = []

        for pdf in all_pdfs:
            slug = pdf['slug']
            try:
                embeddings, _ = self.milvus_client.load_embeddings(slug)
                # Average pooling
                doc_emb = embeddings.mean(axis=0)
                doc_embeddings.append(doc_emb)
                doc_slugs.append(slug)
            except Exception as e:
                print(f"  Warning: Skipping {slug}: {e}")
                continue

        if len(doc_embeddings) < 2:
            print("⚠️  Not enough documents for clustering")
            return {}

        doc_embeddings_matrix = np.vstack(doc_embeddings)

        # 2. K-Means clustering
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        kmeans = KMeans(n_clusters=min(n_topics, len(doc_embeddings)), random_state=42)
        cluster_labels = kmeans.fit_predict(doc_embeddings_matrix)

        # 3. Calculate silhouette score
        silhouette = silhouette_score(doc_embeddings_matrix, cluster_labels)

        # 4. Build results
        topics = {}
        for topic_id in range(n_topics):
            docs_in_topic = [doc_slugs[i] for i, label in enumerate(cluster_labels) if label == topic_id]
            topics[f"topic_{topic_id}"] = {
                'topic_id': topic_id,
                'num_documents': len(docs_in_topic),
                'documents': docs_in_topic,
                'centroid': kmeans.cluster_centers_[topic_id].tolist() if topic_id < len(kmeans.cluster_centers_) else None
            }

        return {
            'topics': topics,
            'silhouette_score': float(silhouette),
            'n_topics': n_topics,
            'n_documents': len(doc_slugs)
        }

    # =========================================================================
    # PHASE 3: CITATION NETWORK
    # =========================================================================

    def build_citation_network(self, all_pdfs: List[Dict]) -> List[Dict[str, Any]]:
        """
        Build citation network based on explicit refs and semantic similarity.

        Args:
            all_pdfs: List of all PDF metadata

        Returns:
            List of citation edges with scores
        """
        citations = []

        # For each PDF, find what it "cites"
        for pdf in all_pdfs:
            slug = pdf['slug']

            # 1. Explicit cross-doc refs
            refs = self.extract_cross_doc_refs(slug)
            resolved_refs = self.resolve_cross_doc_refs(refs)

            # Count references per target
            ref_counts = {}
            for ref in resolved_refs:
                if ref.target_pdf:
                    ref_counts[ref.target_pdf] = ref_counts.get(ref.target_pdf, 0) + 1

            # Create citation edges
            for target_pdf, count in ref_counts.items():
                citations.append({
                    'source': slug,
                    'target': target_pdf,
                    'type': 'explicit',
                    'weight': count,
                    'strength': min(count / 10.0, 1.0),  # Normalize
                })

        return citations


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_relationship_id(source: str, target: str, rel_type: str) -> str:
    """Generate unique ID for a relationship."""
    content = f"{source}_{target}_{rel_type}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


def cross_doc_ref_to_dict(ref: CrossDocRef) -> Dict[str, Any]:
    """Convert CrossDocRef to dictionary."""
    return {
        'source_pdf': ref.source_pdf,
        'source_chunk': ref.source_chunk,
        'mention_text': ref.mention_text,
        'target_document_name': ref.target_document_name,
        'target_pdf': ref.target_pdf,
        'target_section': ref.target_section,
        'target_page': ref.target_page,
        'confidence': ref.confidence,
        'resolution_method': ref.resolution_method,
    }


def semantic_link_to_dict(link: SemanticLink) -> Dict[str, Any]:
    """Convert SemanticLink to dictionary."""
    return {
        'source_pdf': link.source_pdf,
        'source_chunk': link.source_chunk,
        'target_pdf': link.target_pdf,
        'target_chunk': link.target_chunk,
        'similarity': link.similarity,
        'method': link.method,
    }


def version_relation_to_dict(rel: VersionRelation) -> Dict[str, Any]:
    """Convert VersionRelation to dictionary."""
    return {
        'pdf_from': rel.pdf_from,
        'pdf_to': rel.pdf_to,
        'version_from': rel.version_from,
        'version_to': rel.version_to,
        'similarity': rel.similarity,
        'relationship_type': rel.relationship_type,
    }
