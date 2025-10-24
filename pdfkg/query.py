"""
Question-answering functionality using the knowledge graph and embeddings.
"""

import os
from pathlib import Path
from typing import Any

import faiss
import networkx as nx
import numpy as np
import orjson
import pandas as pd
from dotenv import load_dotenv

from pdfkg.embeds import encode_query

# Load .env file
load_dotenv()

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from mistralai import Mistral

    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False


def load_artifacts(pdf_slug: str, storage) -> dict[str, Any]:
    """
    Load pipeline artifacts from storage backend.

    Args:
        pdf_slug: PDF slug identifier
        storage: Storage backend instance

    Returns:
        Dict with keys: chunks, index, sections, figures, tables, graph.
    """
    print(f"DEBUG QUERY: Loading artifacts for PDF: {pdf_slug}")
    artifacts = {}

    # Load chunks
    print(f"DEBUG QUERY: Loading chunks...")
    artifacts["chunks"] = storage.get_chunks(pdf_slug)
    print(f"DEBUG QUERY: Loaded {len(artifacts['chunks'])} chunks")

    # Load FAISS index
    print(f"DEBUG QUERY: Loading FAISS index...")
    artifacts["index"] = storage.load_embeddings(pdf_slug)
    print(f"DEBUG QUERY: FAISS index loaded, dimension: {artifacts['index'].d}, ntotal: {artifacts['index'].ntotal}")

    # Load metadata
    print(f"DEBUG QUERY: Loading metadata...")
    artifacts["sections"] = storage.get_metadata(pdf_slug, "sections") or {}
    artifacts["figures"] = storage.get_metadata(pdf_slug, "figures") or {}
    artifacts["tables"] = storage.get_metadata(pdf_slug, "tables") or {}
    print(f"DEBUG QUERY: Loaded {len(artifacts['sections'])} sections, {len(artifacts['figures'])} figures, {len(artifacts['tables'])} tables")

    # Load graph if available
    if hasattr(storage, 'get_graph'):
        print(f"DEBUG QUERY: Loading graph...")
        nodes, edges = storage.get_graph(pdf_slug)
        print(f"DEBUG QUERY: Reconstructing NetworkX graph from {len(nodes)} nodes and {len(edges)} edges")
        # Reconstruct NetworkX graph
        G = nx.MultiDiGraph()
        for node in nodes:
            node_id = node.get("node_id")
            G.add_node(node_id, **{k: v for k, v in node.items() if k != "node_id"})
        for edge in edges:
            from_id = edge.get("from_id")
            to_id = edge.get("to_id")
            if from_id and to_id:
                G.add_edge(from_id, to_id, **{k: v for k, v in edge.items() if k not in ["from_id", "to_id", "_from", "_to"]})
        artifacts["graph"] = G
        print(f"DEBUG QUERY: Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    else:
        print(f"DEBUG QUERY: Storage backend doesn't support graphs, using empty graph")
        artifacts["graph"] = nx.MultiDiGraph()

    return artifacts


def retrieve_chunks(
    question: str,
    artifacts: dict[str, Any],
    model_name: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve most relevant chunks for a question using FAISS.

    Args:
        question: User question.
        artifacts: Loaded artifacts dict.
        model_name: Embedding model name.
        top_k: Number of chunks to retrieve.

    Returns:
        List of chunk dicts with similarity scores.
    """
    print(f"DEBUG QUERY: Retrieving chunks for question: '{question}'")
    print(f"DEBUG QUERY: Using model: {model_name}, top_k: {top_k}")

    index = artifacts["index"]
    expected_dim = getattr(index, "d", None)
    if expected_dim is not None:
        print(f"DEBUG QUERY: FAISS index expects dimension: {expected_dim}")

    # Encode question
    print(f"DEBUG QUERY: Encoding question...")
    query_embedding = encode_query(question, model_name)
    print(f"DEBUG QUERY: Query embedding shape: {query_embedding.shape}")

    if expected_dim is not None and query_embedding.shape[1] != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: model '{model_name}' produced "
            f"{query_embedding.shape[1]} dimensions but FAISS index expects {expected_dim}. "
            "Reprocess the PDF with the desired embedding model or configure the chat to use "
            "the same model that was used during ingestion."
        )

    # Search FAISS index
    print(f"DEBUG QUERY: Searching FAISS index...")
    distances, indices = index.search(query_embedding, top_k)
    print(f"DEBUG QUERY: FAISS search complete. Found {len(indices[0])} results")

    # Get chunks with scores
    results = []
    chunks = artifacts["chunks"]
    for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
        if idx < len(chunks):
            chunk = chunks[idx].copy()
            chunk["similarity_score"] = float(score)
            chunk["rank"] = i + 1
            results.append(chunk)
            print(f"DEBUG QUERY:   Chunk {i+1}: idx={idx}, score={score:.3f}, section={chunk.get('section_id', 'N/A')}, page={chunk.get('page', 'N/A')}")

    print(f"DEBUG QUERY: Retrieved {len(results)} chunks")
    return results


def retrieve_chunks_global(
    question: str,
    model_name: str,
    top_k: int,
    storage,
) -> list[dict]:
    """
    Retrieve most relevant chunks across ALL PDFs using Milvus global search.

    Args:
        question: User question.
        model_name: Embedding model name.
        top_k: Number of chunks to retrieve.
        storage: Storage backend with Milvus client.

    Returns:
        List of chunk dicts with similarity scores and PDF info.
    """
    print(f"DEBUG QUERY: Retrieving chunks GLOBALLY across all PDFs")
    print(f"DEBUG QUERY: Question: '{question}'")
    print(f"DEBUG QUERY: Using model: {model_name}, top_k: {top_k}")

    # Check if Milvus is available
    if not hasattr(storage, 'milvus_client') or storage.milvus_client is None:
        raise RuntimeError("Milvus client not available for global search")

    # Encode question
    print(f"DEBUG QUERY: Encoding question...")
    query_embedding = encode_query(question, model_name)
    print(f"DEBUG QUERY: Query embedding shape: {query_embedding.shape}")

    # Perform global search in Milvus
    print(f"DEBUG QUERY: Performing global search in Milvus...")
    search_results = storage.milvus_client.search_global(
        query_embedding=query_embedding,
        top_k=top_k
    )
    print(f"DEBUG QUERY: Global search complete. Found {len(search_results)} results")

    # Fetch chunks from ArangoDB
    results = []
    for i, result in enumerate(search_results):
        pdf_slug = result['pdf_slug']
        chunk_index = result['chunk_index']
        distance = result['distance']

        print(f"DEBUG QUERY:   Result {i+1}: pdf={pdf_slug}, chunk_idx={chunk_index}, score={distance:.3f}")

        # Get chunk from ArangoDB
        try:
            chunks = storage.db_client.get_chunks(pdf_slug)
            if chunk_index < len(chunks):
                chunk = chunks[chunk_index].copy()
                chunk["similarity_score"] = distance
                chunk["rank"] = i + 1
                chunk["pdf_slug"] = pdf_slug  # Add PDF slug to chunk

                # Get PDF metadata for display
                pdf_info = storage.db_client.get_pdf(pdf_slug)
                if pdf_info:
                    chunk["pdf_filename"] = pdf_info.get("filename", pdf_slug)
                else:
                    chunk["pdf_filename"] = pdf_slug

                results.append(chunk)
            else:
                print(f"DEBUG QUERY:   Warning: chunk_index {chunk_index} out of range for PDF {pdf_slug}")
        except Exception as e:
            print(f"DEBUG QUERY:   Error fetching chunk from {pdf_slug}: {e}")
            continue

    print(f"DEBUG QUERY: Retrieved {len(results)} chunks from {len(set(r['pdf_slug'] for r in results))} PDFs")
    return results


def find_related_nodes(
    chunk_ids: list[str], graph: nx.MultiDiGraph, max_depth: int = 1
) -> dict[str, list[str]]:
    """
    Find related nodes (figures, tables, sections) for given chunks.

    Args:
        chunk_ids: List of chunk IDs.
        graph: Knowledge graph.
        max_depth: Maximum traversal depth.

    Returns:
        Dict with keys: figures, tables, sections (lists of node IDs).
    """
    related = {"figures": set(), "tables": set(), "sections": set()}

    for chunk_id in chunk_ids:
        if not graph.has_node(chunk_id):
            continue

        # Find outgoing REFERS_TO edges
        for _, target, data in graph.out_edges(chunk_id, data=True):
            if data.get("type") == "REFERS_TO":
                node_type = graph.nodes[target].get("type")
                if node_type == "Figure":
                    related["figures"].add(target)
                elif node_type == "Table":
                    related["tables"].add(target)
                elif node_type == "Section":
                    related["sections"].add(target)

    return {k: list(v) for k, v in related.items()}


def generate_answer_gemini(question: str, context_chunks: list[dict]) -> str:
    """
    Generate answer using Gemini.

    Args:
        question: User question.
        context_chunks: Retrieved chunks with metadata.

    Returns:
        Generated answer.
    """
    if not GEMINI_AVAILABLE:
        raise RuntimeError("google-generativeai not installed")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)

    # Get model name from environment or use default
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

    # Build context (include PDF filename if available for global search)
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        pdf_info = f" PDF: {chunk['pdf_filename']}" if 'pdf_filename' in chunk else ""
        context_parts.append(
            f"[Chunk {i}]{pdf_info} (Section: {chunk['section_id']}, Page: {chunk['page']})\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant answering questions about a technical manual.

Question: {question}

Context from the manual:
{context}

Instructions:
- Answer the question based ONLY on the provided context
- Be specific and cite section/page references when possible
- If the context doesn't contain enough information to answer, say so
- Keep your answer concise and technical

Answer:"""

    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)

    return response.text


def generate_answer_mistral(question: str, context_chunks: list[dict]) -> str:
    """
    Generate answer using Mistral AI.

    Args:
        question: User question.
        context_chunks: Retrieved chunks with metadata.

    Returns:
        Generated answer.
    """
    if not MISTRAL_AVAILABLE:
        raise RuntimeError(
            "mistralai package not installed. Install with: pip install mistralai"
        )

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not set in .env")

    # Get model name from environment or use default
    model_name = os.getenv("MISTRAL_MODEL", "mistral-large-latest")

    # Initialize Mistral client
    client = Mistral(api_key=api_key)

    # Build context (include PDF filename if available for global search)
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        pdf_info = f" PDF: {chunk['pdf_filename']}" if 'pdf_filename' in chunk else ""
        context_parts.append(
            f"[Chunk {i}]{pdf_info} (Section: {chunk['section_id']}, Page: {chunk['page']})\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    # Build prompt
    prompt = f"""You are a helpful assistant answering questions about a technical manual.

Question: {question}

Context from the manual:
{context}

Instructions:
- Answer the question based ONLY on the provided context
- Be specific and cite section/page references when possible
- If the context doesn't contain enough information to answer, say so
- Keep your answer concise and technical

Answer:"""

    # Generate response
    response = client.chat.complete(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.choices[0].message.content


def format_simple_answer(question: str, context_chunks: list[dict]) -> str:
    """
    Format a simple answer without LLM (just show relevant chunks).

    Args:
        question: User question.
        context_chunks: Retrieved chunks with metadata.

    Returns:
        Formatted answer showing relevant chunks.
    """
    lines = [f"Question: {question}\n"]
    lines.append("Relevant information from the manual:\n")

    for i, chunk in enumerate(context_chunks, 1):
        pdf_info = f" [PDF: {chunk['pdf_filename']}]" if 'pdf_filename' in chunk else ""
        lines.append(
            f"\n[{i}]{pdf_info} Section {chunk['section_id']}, Page {chunk['page']} (similarity: {chunk['similarity_score']:.3f})"
        )
        lines.append(f"{chunk['text'][:500]}{'...' if len(chunk['text']) > 500 else ''}\n")

    return "\n".join(lines)


def answer_question(
    question: str,
    pdf_slug: str = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 5,
    llm_provider: str = "none",
    storage = None,
) -> dict[str, Any]:
    """
    Answer a question using the knowledge graph.

    Args:
        question: User question.
        pdf_slug: PDF slug identifier. If None, searches across ALL PDFs (global search).
        model_name: Embedding model name.
        top_k: Number of chunks to retrieve.
        llm_provider: LLM provider to use ("none", "gemini", or "mistral").
        storage: Storage backend instance (optional, will create if None).

    Returns:
        Dict with keys: question, answer, sources, related.
    """
    import time
    start_time = time.time()

    print(f"\nDEBUG QUERY: ========== answer_question START ==========")
    print(f"DEBUG QUERY: Question: {question}")
    print(f"DEBUG QUERY: PDF slug: {pdf_slug}")
    print(f"DEBUG QUERY: Model: {model_name}")
    print(f"DEBUG QUERY: Top K: {top_k}")
    print(f"DEBUG QUERY: LLM Provider: {llm_provider}")

    # Get storage backend if not provided
    if storage is None:
        print(f"DEBUG QUERY: No storage provided, getting default backend...")
        from pdfkg.storage import get_storage_backend
        storage = get_storage_backend()
    else:
        print(f"DEBUG QUERY: Using provided storage backend: {type(storage).__name__}")

    # Check if this is a global search (all PDFs) or single PDF search
    is_global_search = pdf_slug is None

    if is_global_search:
        print(f"DEBUG QUERY: GLOBAL SEARCH MODE - Searching across ALL PDFs")

        # Use default embedding model for global search
        # TODO: Could auto-detect from first PDF or use env variable
        print(f"DEBUG QUERY: Using embedding model: {model_name}")

        # Retrieve chunks globally using Milvus
        print(f"DEBUG QUERY: Retrieving relevant chunks globally...")
        chunks = retrieve_chunks_global(question, model_name, top_k, storage)

        # Create empty artifacts dict (no single PDF artifacts for global search)
        artifacts = {"graph": nx.MultiDiGraph()}

    else:
        print(f"DEBUG QUERY: SINGLE PDF SEARCH MODE - Searching in PDF: {pdf_slug}")

        # Original single-PDF logic
        stored_model = None
        stored_dim = None
        pdf_info = None
        try:
            pdf_info = storage.get_pdf_metadata(pdf_slug)
            if pdf_info:
                print("DEBUG QUERY: Loaded PDF metadata for embedding lookup")
        except Exception as exc:
            print(f"DEBUG QUERY: Failed to load PDF metadata: {exc}")

        if pdf_info:
            metadata = pdf_info.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}

            stored_model = metadata.get("embedding_model") or pdf_info.get("embedding_model")
            stored_dim = metadata.get("embedding_dim") or pdf_info.get("embedding_dim")

            if stored_model:
                print(f"DEBUG QUERY: Detected stored embedding model: {stored_model}")
                if stored_model != model_name:
                    print(
                        "DEBUG QUERY: Overriding requested embedding model with stored model to "
                        "match FAISS index"
                    )
                    model_name = stored_model
            else:
                print("DEBUG QUERY: No stored embedding model found in metadata")

            if stored_dim:
                print(f"DEBUG QUERY: Stored embedding dimension: {stored_dim}")
        else:
            print("DEBUG QUERY: PDF metadata not available; using requested embedding model")

        # Dimension to model mapping (common models)
        DIM_TO_MODEL = {
            384: "sentence-transformers/all-MiniLM-L6-v2",
            768: "sentence-transformers/all-mpnet-base-v2",
            1024: "BAAI/bge-large-en-v1.5",
        }

        # Load artifacts
        print(f"DEBUG QUERY: Loading artifacts...")
        artifacts = load_artifacts(pdf_slug, storage)

        index_dim = getattr(artifacts.get("index", None), "d", None)
        if stored_dim and index_dim and stored_dim != index_dim:
            print(
                f"DEBUG QUERY: Stored embedding dimension ({stored_dim}) does not match FAISS index "
                f"dimension ({index_dim}). Using index dimension for validation."
            )
            stored_dim = index_dim
        elif not stored_dim and index_dim:
            print(f"DEBUG QUERY: Using FAISS index dimension {index_dim} for validation")
            stored_dim = index_dim

        # If no model was stored but we have a dimension, auto-select model
        if not stored_model and stored_dim and stored_dim in DIM_TO_MODEL:
            auto_model = DIM_TO_MODEL[stored_dim]
            print(f"DEBUG QUERY: No stored model found, auto-selecting based on dimension {stored_dim}: {auto_model}")
            if auto_model != model_name:
                print(f"DEBUG QUERY: Overriding requested model '{model_name}' with auto-detected '{auto_model}'")
                model_name = auto_model

        # Retrieve relevant chunks
        print(f"DEBUG QUERY: Retrieving relevant chunks...")
        chunks = retrieve_chunks(question, artifacts, model_name, top_k)

    # Find related nodes
    print(f"DEBUG QUERY: Finding related nodes...")
    chunk_ids = [c.get("chunk_id") or c.get("id") for c in chunks]
    print(f"DEBUG QUERY: Chunk IDs: {chunk_ids}")
    related = find_related_nodes(chunk_ids, artifacts["graph"])
    print(f"DEBUG QUERY: Related: {related}")

    # Generate answer
    print(f"DEBUG QUERY: Generating answer...")
    llm_model = "none"

    if llm_provider == "gemini":
        print(f"DEBUG QUERY: Using Gemini for answer generation")
        answer = generate_answer_gemini(question, chunks)
        llm_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    elif llm_provider == "mistral":
        print(f"DEBUG QUERY: Using Mistral for answer generation")
        answer = generate_answer_mistral(question, chunks)
        llm_model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
    else:
        print(f"DEBUG QUERY: Using simple answer format (no LLM)")
        answer = format_simple_answer(question, chunks)
        llm_provider = "none"

    print(f"DEBUG QUERY: Answer generated successfully")

    # Calculate response time
    response_time_ms = (time.time() - start_time) * 1000
    print(f"DEBUG QUERY: Response time: {response_time_ms:.2f}ms")

    # Save Q&A interaction to database (if using ArangoDB storage)
    if hasattr(storage, 'db_client') and hasattr(storage.db_client, 'save_qa_interaction'):
        try:
            print(f"DEBUG QUERY: Saving Q&A interaction to database...")
            # Prepare sources for storage (limit data size)
            sources_for_storage = [
                {
                    "chunk_id": c.get("chunk_id") or c.get("id"),
                    "section_id": c.get("section_id"),
                    "page": c.get("page"),
                    "similarity_score": c.get("similarity_score"),
                    "rank": c.get("rank"),
                }
                for c in chunks
            ]

            storage.db_client.save_qa_interaction(
                question=question,
                answer=answer,
                pdf_slug=pdf_slug,
                llm_provider=llm_provider,
                llm_model=llm_model,
                embed_model=model_name,
                top_k=top_k,
                sources=sources_for_storage,
                related_items=related,
                response_time_ms=response_time_ms,
            )
            print(f"DEBUG QUERY: Q&A interaction saved successfully")
        except Exception as e:
            print(f"DEBUG QUERY: Warning - Could not save Q&A interaction: {e}")

    print(f"DEBUG QUERY: ========== answer_question END ==========\n")

    return {
        "question": question,
        "answer": answer,
        "sources": chunks,
        "related": related,
    }
