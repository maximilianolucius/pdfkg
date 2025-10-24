"""
PDF management utilities for multi-PDF platform.

This module provides the core PDF ingestion pipeline shared by CLI and web interfaces.
"""

import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Optional, Callable, Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
import orjson
import networkx as nx


class PDFManager:
    """Manage multiple PDFs and their processed artifacts."""

    def __init__(self, base_dir: Path = Path("data")):
        self.base_dir = base_dir
        self.input_dir = base_dir / "input"
        self.output_dir = base_dir / "out"
        self.registry_file = self.output_dir / "processed_pdfs.json"

        # Create directories
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_pdf_slug(self, filename: str) -> str:
        """Convert filename to slug for directory names."""
        # Remove .pdf extension and sanitize
        slug = Path(filename).stem
        # Replace spaces and special chars with underscores
        slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in slug)
        return slug.lower()

    def register_pdf(
        self,
        filename: str,
        num_pages: int,
        num_chunks: int,
        num_sections: int,
        *,
        slug: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict:
        """Register a processed PDF in the registry."""
        registry = self.load_registry()

        pdf_slug = slug or self.get_pdf_slug(filename)
        pdf_info = {
            "filename": filename,
            "slug": pdf_slug,
            "processed_date": datetime.now().isoformat(),
            "num_pages": num_pages,
            "num_chunks": num_chunks,
            "num_sections": num_sections,
            "output_dir": str(self.output_dir / pdf_slug),
            "metadata": metadata or {},
        }

        registry[pdf_slug] = pdf_info
        self.save_registry(registry)
        return pdf_info

    def load_registry(self) -> dict:
        """Load the PDF registry."""
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                return json.load(f)
        return {}

    def save_registry(self, registry: dict) -> None:
        """Save the PDF registry."""
        with open(self.registry_file, "w") as f:
            json.dump(registry, f, indent=2)

    def get_pdf_info(self, slug: str) -> Optional[dict]:
        """Get info for a specific PDF."""
        registry = self.load_registry()
        return registry.get(slug)

    def list_pdfs(self) -> list[dict]:
        """List all processed PDFs."""
        registry = self.load_registry()
        return list(registry.values())

    def get_pdf_output_dir(self, slug: str) -> Path:
        """Get output directory for a PDF."""
        return self.output_dir / slug

    def get_pdf_input_path(self, filename: str) -> Path:
        """Get input path for a PDF."""
        return self.input_dir / filename

    def pdf_exists(self, slug: str) -> bool:
        """Check if a PDF has been processed."""
        registry = self.load_registry()
        return slug in registry

    def delete_pdf(self, slug: str) -> bool:
        """Delete a PDF and its artifacts."""
        registry = self.load_registry()
        if slug not in registry:
            return False

        # Remove from registry
        pdf_info = registry.pop(slug)
        self.save_registry(registry)

        # Optionally delete files (commented out for safety)
        # output_dir = Path(pdf_info["output_dir"])
        # if output_dir.exists():
        #     shutil.rmtree(output_dir)

        return True


class IngestionResult:
    """Container for ingestion pipeline results."""

    def __init__(
        self,
        pdf_slug: str,
        original_filename: str,
        pages: List[Dict],
        sections: Dict,
        toc: List[Dict],
        chunks: List,
        embeddings,
        figures: Dict,
        tables: Dict,
        mentions: List,
        graph: nx.MultiDiGraph,
        was_cached: bool = False,
        aas_classification: Optional[Dict] = None,
    ):
        self.pdf_slug = pdf_slug
        self.original_filename = original_filename
        self.pages = pages
        self.sections = sections
        self.toc = toc
        self.chunks = chunks
        self.embeddings = embeddings
        self.figures = figures
        self.tables = tables
        self.mentions = mentions
        self.graph = graph
        self.was_cached = was_cached
        self.aas_classification = aas_classification

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics."""
        return {
            "pdf_slug": self.pdf_slug,
            "filename": self.original_filename,
            "num_pages": len(self.pages),
            "num_sections": len(self.sections),
            "num_chunks": len(self.chunks),
            "num_figures": len(self.figures),
            "num_tables": len(self.tables),
            "num_mentions": len(self.mentions),
            "num_resolved_mentions": sum(1 for m in self.mentions if m.target_id),
            "num_graph_nodes": self.graph.number_of_nodes(),
            "num_graph_edges": self.graph.number_of_edges(),
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "was_cached": self.was_cached,
            "aas_classification": self.aas_classification,
        }


def ingest_pdf(
    pdf_path: Path,
    storage=None,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_tokens: int = 500,
    use_gemini: bool = False,
    gemini_pages: str = "",
    force_reprocess: bool = False,
    save_to_db: bool = True,
    save_files: bool = True,
    output_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> IngestionResult:
    """
    Unified PDF ingestion pipeline used by both CLI and web interface.

    This function implements the complete 13-step ingestion pipeline:
    1. PDF Loading & Validation
    2. Slug Generation
    3. Cache Check (optional skip with force_reprocess)
    4. Page Extraction
    5. Table of Contents Extraction
    6. Section Tree Building
    7. Text Chunking
    8. Embedding Generation
    9. FAISS Index Building
    10. Figure & Table Indexing
    11. Cross-Reference Extraction & Resolution
    12. Gemini Visual Analysis (optional)
    13. Knowledge Graph Construction

    Args:
        pdf_path: Path to PDF file
        storage: Storage backend (ArangoDB or FileStorage). If None, uses file-only mode.
        embed_model: Sentence-transformers model name
        max_tokens: Maximum tokens per chunk
        use_gemini: Enable Gemini visual cross-reference extraction
        gemini_pages: Page ranges for Gemini (e.g., "1-10,30-40"). If empty and use_gemini=True, processes all pages.
        force_reprocess: Reprocess even if PDF already exists in cache
        save_to_db: Save to database backend (requires storage)
        save_files: Export legacy file formats
        output_dir: Output directory for file exports (default: data/out/{slug})
        progress_callback: Function(progress: float, desc: str) for progress updates

    Returns:
        IngestionResult containing all processed artifacts

    Raises:
        FileNotFoundError: If PDF file doesn't exist
        ValueError: If invalid parameters provided
    """
    from pdfkg.parse_pdf import load_pdf, extract_pages, extract_toc
    from pdfkg.topology import build_section_tree
    from pdfkg.chunking import build_chunks
    from pdfkg.embeds import embed_chunks, build_faiss_index
    from pdfkg.figtables import index_figures_tables
    from pdfkg.xrefs import extract_mentions, resolve_mentions
    from pdfkg.graph import build_graph, export_graph
    from pdfkg.report import generate_report
    from pdfkg.aas_classifier import classify_single_pdf_submodels

    # Optional Gemini import
    gemini_available = False
    if use_gemini or os.getenv("GEMINI_API_KEY"):
        try:
            from pdfkg.gemini_helpers import gemini_extract_crossrefs, merge_gemini_crossrefs
            gemini_available = True
        except ImportError:
            pass

    def progress(pct: float, desc: str):
        """Internal progress helper."""
        if progress_callback:
            progress_callback(pct, desc)

    # Validate inputs
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Generate slug and setup paths
    original_filename = pdf_path.name
    pdf_slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in pdf_path.stem).lower()

    if output_dir is None:
        output_dir = Path("data/out") / pdf_slug
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to data/input/ for persistence
    input_dir = Path("data/input")
    input_dir.mkdir(parents=True, exist_ok=True)
    persistent_pdf_path = input_dir / original_filename
    if not persistent_pdf_path.exists():
        shutil.copy(pdf_path, persistent_pdf_path)

    progress(0.05, "Checking cache...")

    # Check cache (if not forcing reprocess)
    if not force_reprocess and storage and save_to_db:
        pdf_info = storage.get_pdf_metadata(pdf_slug)
        if pdf_info:
            # PDF already processed - load from cache
            progress(0.1, "Loading from cache...")

            # Load artifacts from storage
            chunks_data = storage.load_chunks(pdf_slug)
            faiss_index = storage.load_embeddings(pdf_slug)

            # Convert FAISS index back to numpy array for compatibility
            embeddings = np.zeros((faiss_index.ntotal, faiss_index.d), dtype=np.float32)
            if faiss_index.ntotal > 0:
                faiss_index.reconstruct_n(0, faiss_index.ntotal, embeddings)

            metadata = storage.load_all_metadata(pdf_slug)

            # Reconstruct chunks (need to create proper Chunk objects)
            from pdfkg.chunking import Chunk
            chunks = [
                Chunk(
                    id=c['chunk_id'],
                    section_id=c['section_id'],
                    page=c['page'],
                    text=c['text']
                )
                for c in chunks_data
            ]

            # Reconstruct mentions
            from pdfkg.xrefs import Mention
            mentions = [
                Mention(
                    source_chunk_id=m['source_chunk_id'],
                    kind=m['kind'],
                    raw_text=m['raw_text'],
                    target_hint=m.get('target_hint'),
                    target_id=m.get('target_id')
                )
                for m in metadata.get('mentions', [])
            ]

            # Load graph
            if hasattr(storage, 'load_graph'):
                graph = storage.load_graph(pdf_slug)
            else:
                # Build minimal graph
                from pdfkg.graph import build_graph
                graph = build_graph(
                    doc_id="document",
                    pages=[{"page": i+1} for i in range(pdf_info['num_pages'])],
                    sections=metadata.get('sections', {}),
                    chunks=chunks,
                    mentions=mentions,
                    figures=metadata.get('figures', {}),
                    tables=metadata.get('tables', {}),
                )

            progress(1.0, "Loaded from cache")

            return IngestionResult(
                pdf_slug=pdf_slug,
                original_filename=original_filename,
                pages=[{"page": i+1} for i in range(pdf_info['num_pages'])],
                sections=metadata.get('sections', {}),
                toc=metadata.get('toc', []),
                chunks=chunks,
                embeddings=embeddings,
                figures=metadata.get('figures', {}),
                tables=metadata.get('tables', {}),
                mentions=mentions,
                graph=graph,
                was_cached=True,
                aas_classification=metadata.get('aas_classification'),
            )

    # Step 1: Load PDF
    progress(0.1, "Loading PDF...")
    doc = load_pdf(pdf_path)

    # Step 4: Extract pages
    progress(0.2, "Extracting pages...")
    pages = extract_pages(doc)

    # Step 5: Extract ToC
    progress(0.25, "Extracting table of contents...")
    toc = extract_toc(doc)

    # Step 6: Build section tree
    progress(0.3, "Building section tree...")
    sections = build_section_tree(toc)

    # Step 7: Chunk text
    progress(0.35, "Chunking text...")
    chunks = build_chunks(pages, sections, max_tokens=max_tokens)

    # Step 8: Generate embeddings
    progress(0.45, "Generating embeddings (this may take a minute)...")
    embeddings = embed_chunks(chunks, model_name=embed_model)

    # Step 9: Build FAISS index
    progress(0.6, "Building search index...")
    if storage and save_to_db:
        # Extract chunk IDs for Milvus
        chunk_ids = [c.id for c in chunks]
        storage.save_embeddings(pdf_slug, embeddings, chunk_ids=chunk_ids)
    else:
        # Save to file
        index = build_faiss_index(embeddings)
        faiss.write_index(index, str(output_dir / "index.faiss"))

    # Step 10: Index figures and tables
    progress(0.65, "Indexing figures and tables...")
    figures, tables = index_figures_tables(pages)

    # Step 11: Extract and resolve cross-references
    progress(0.7, "Extracting cross-references...")
    all_mentions = []
    for chunk in chunks:
        all_mentions.extend(extract_mentions(chunk))

    progress(0.75, "Resolving cross-references...")
    all_mentions = resolve_mentions(
        all_mentions, sections, figures, tables, n_pages=len(pages)
    )

    # Step 12: Optional Gemini visual analysis
    if use_gemini:
        if not gemini_available:
            progress(0.77, "Warning: Gemini not available, skipping visual analysis")
        elif not os.getenv("GEMINI_API_KEY"):
            progress(0.77, "Warning: GEMINI_API_KEY not set, skipping visual analysis")
        else:
            progress(0.78, "Running Gemini visual analysis...")

            # Determine page ranges
            if gemini_pages:
                # Parse user-specified ranges
                page_ranges = parse_page_ranges(gemini_pages)
            else:
                # Process all pages by default
                page_ranges = [(1, len(pages))]

            gemini_results = {"cross_references": []}
            for start, end in page_ranges:
                progress(0.78, f"Gemini processing pages {start}-{end}...")
                result = gemini_extract_crossrefs(persistent_pdf_path, start, end)
                gemini_results["cross_references"].extend(
                    result.get("cross_references", [])
                )

            progress(0.8, "Merging Gemini results...")
            all_mentions = merge_gemini_crossrefs(
                all_mentions, gemini_results, sections, figures, tables
            )

    # Step 13: Build knowledge graph
    progress(0.85, "Building knowledge graph...")
    graph = build_graph(
        doc_id="document",
        pages=pages,
        sections=sections,
        chunks=chunks,
        mentions=all_mentions,
        figures=figures,
        tables=tables,
    )

    # Save to database
    if storage and save_to_db:
        progress(0.9, "Saving to database...")

        # Register PDF first
        storage.save_pdf_metadata(
            slug=pdf_slug,
            filename=original_filename,
            num_pages=len(pages),
            num_chunks=len(chunks),
            num_sections=len(sections),
            num_figures=len(figures),
            num_tables=len(tables),
            metadata={
                "embedding_model": embed_model,
                "embedding_dim": int(embeddings.shape[1]),
                "max_tokens": max_tokens,
                "used_gemini": use_gemini,
            },
        )

        # Save chunks
        chunks_data = [
            {"chunk_id": c.id, "section_id": c.section_id, "page": c.page, "text": c.text}
            for c in chunks
        ]
        storage.save_chunks(pdf_slug, chunks_data)

        # Save graph
        if hasattr(storage, 'save_graph'):
            nodes = []
            for node_id, attrs in graph.nodes(data=True):
                node_doc = {
                    "node_id": node_id,
                    "type": attrs.get("type", "Unknown"),
                    "label": attrs.get("label", "")
                }
                for k, v in attrs.items():
                    if k not in ["type", "label"] and v is not None:
                        node_doc[k] = v
                nodes.append(node_doc)

            edges = []
            for u, v, attrs in graph.edges(data=True):
                edge_doc = {
                    "from_id": u,
                    "to_id": v,
                    "type": attrs.get("type", "EDGE")
                }
                for k, val in attrs.items():
                    if k not in ["type"] and val is not None:
                        edge_doc[k] = val
                edges.append(edge_doc)

            storage.save_graph(pdf_slug, nodes, edges)

        # Save metadata
        storage.save_metadata(pdf_slug, "sections", sections)
        storage.save_metadata(pdf_slug, "toc", toc)
        storage.save_metadata(pdf_slug, "mentions", [
            {
                "source_chunk_id": m.source_chunk_id,
                "kind": m.kind,
                "raw_text": m.raw_text,
                "target_hint": m.target_hint,
                "target_id": m.target_id,
            }
            for m in all_mentions
        ])
        storage.save_metadata(pdf_slug, "figures", figures)
        storage.save_metadata(pdf_slug, "tables", tables)

    # New Step 14: AAS Submodel Classification
    aas_classification_result = None
    if storage and save_to_db and os.getenv("GEMINI_API_KEY"):
        progress(0.92, "Classifying AAS submodels...")
        aas_classification_result = classify_single_pdf_submodels(
            storage=storage,
            pdf_slug=pdf_slug,
            llm_provider="gemini"
        )
        if aas_classification_result:
            storage.save_metadata(pdf_slug, "aas_classification", aas_classification_result)
            progress(0.94, "AAS classification saved")

    # Export files
    if save_files:
        progress(0.95, "Exporting files...")

        # Export graph
        export_graph(graph, output_dir)

        # Export data
        chunks_df = pd.DataFrame([
            {"id": c.id, "section_id": c.section_id, "page": c.page, "text": c.text}
            for c in chunks
        ])
        chunks_df.to_parquet(output_dir / "chunks.parquet", index=False)

        mentions_df = pd.DataFrame([
            {
                "source_chunk_id": m.source_chunk_id,
                "kind": m.kind,
                "raw_text": m.raw_text,
                "target_hint": m.target_hint,
                "target_id": m.target_id,
            }
            for m in all_mentions
        ])
        mentions_df.to_parquet(output_dir / "mentions.parquet", index=False)

        (output_dir / "sections.json").write_bytes(
            orjson.dumps(sections, option=orjson.OPT_INDENT_2)
        )
        (output_dir / "toc.json").write_bytes(
            orjson.dumps(toc, option=orjson.OPT_INDENT_2)
        )

        # Generate report
        generate_report(
            output_dir / "report.md",
            sections, chunks, all_mentions, figures, tables
        )

    progress(1.0, "Complete!")

    return IngestionResult(
        pdf_slug=pdf_slug,
        original_filename=original_filename,
        pages=pages,
        sections=sections,
        toc=toc,
        chunks=chunks,
        embeddings=embeddings,
        figures=figures,
        tables=tables,
        mentions=all_mentions,
        graph=graph,
        was_cached=False,
        aas_classification=aas_classification_result,
    )


def parse_page_ranges(ranges_str: str) -> List[Tuple[int, int]]:
    """
    Parse page ranges like '1-10,30-40' into list of (start, end) tuples.

    Args:
        ranges_str: Comma-separated ranges (e.g., "1-10,30-40,50")

    Returns:
        List of (start, end) inclusive tuples
    """
    ranges = []
    for part in ranges_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            ranges.append((int(start), int(end)))
        else:
            page = int(part)
            ranges.append((page, page))
    return ranges


def auto_ingest_directory(
    input_dir: Path = Path("data/input"),
    storage=None,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_tokens: int = 500,
    use_gemini: bool = False,
    save_to_db: bool = True,
    save_files: bool = True,
    progress_callback: Optional[Callable[[str, str], None]] = None,
    force_reprocess: bool = False,
) -> Dict[str, List[str]]:
    """
    Automatically ingest all PDFs from a directory, skipping already-processed ones.

    Args:
        input_dir: Directory containing PDF files
        storage: Storage backend (required if save_to_db=True)
        embed_model: Sentence-transformers model name
        max_tokens: Maximum tokens per chunk
        use_gemini: Enable Gemini visual analysis for all PDFs
        save_to_db: Save to database backend
        save_files: Export legacy file formats
        progress_callback: Function(pdf_name: str, status: str) for progress updates
        force_reprocess: Re-ingest even if entries already exist in storage

    Returns:
        Dictionary with 'processed', 'skipped', and 'failed' lists of filenames
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        input_dir.mkdir(parents=True, exist_ok=True)
        return {'processed': [], 'skipped': [], 'failed': []}

    # Find all PDF files
    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        return {'processed': [], 'skipped': [], 'failed': []}

    results = {
        'processed': [],
        'skipped': [],
        'failed': []
    }

    for pdf_path in pdf_files:
        pdf_name = pdf_path.name

        try:
            # Generate slug to check if already processed
            pdf_slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in pdf_path.stem).lower()

            # Check if already in database
            already_processed = False
            if not force_reprocess and storage and save_to_db:
                pdf_info = storage.get_pdf_metadata(pdf_slug)
                if pdf_info:
                    already_processed = True

            if already_processed:
                if progress_callback:
                    progress_callback(pdf_name, "skipped (already processed)")
                results['skipped'].append(pdf_name)
                continue

            # Process the PDF
            if progress_callback:
                progress_callback(pdf_name, "processing...")

            def pdf_progress(pct: float, desc: str):
                if progress_callback:
                    progress_callback(pdf_name, f"{desc} ({int(pct*100)}%)")

            result = ingest_pdf(
                pdf_path=pdf_path,
                storage=storage,
                embed_model=embed_model,
                max_tokens=max_tokens,
                use_gemini=use_gemini,
                gemini_pages="",  # Process all pages if Gemini enabled
                force_reprocess=force_reprocess,
                save_to_db=save_to_db,
                save_files=save_files,
                output_dir=None,
                progress_callback=pdf_progress,
            )

            if progress_callback:
                progress_callback(pdf_name, "✓ completed")
            results['processed'].append(pdf_name)

        except Exception as e:
            if progress_callback:
                progress_callback(pdf_name, f"✗ failed: {str(e)}")
            results['failed'].append(pdf_name)
            # Continue processing other PDFs even if one fails
            continue

    return results
