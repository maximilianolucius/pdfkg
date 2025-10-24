"""
Unified storage interface supporting both file-based and ArangoDB storage.
"""

import os
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
import orjson
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class MilvusIndexWrapper:
    """
    Wrapper around Milvus client to provide FAISS-like search interface.

    This allows query.py to use Milvus transparently without code changes.
    """

    def __init__(self, milvus_client, pdf_slug: str):
        """
        Initialize wrapper.

        Args:
            milvus_client: MilvusClient instance
            pdf_slug: PDF identifier
        """
        self.milvus_client = milvus_client
        self.pdf_slug = pdf_slug
        self._embeddings = None
        self._chunk_ids = None
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy load embeddings from Milvus."""
        if not self._loaded:
            self._embeddings, self._chunk_ids = self.milvus_client.load_embeddings(self.pdf_slug)
            self._loaded = True

    @property
    def d(self) -> int:
        """Get embedding dimension (FAISS-like interface)."""
        self._ensure_loaded()
        return self._embeddings.shape[1]

    @property
    def ntotal(self) -> int:
        """Get total number of vectors (FAISS-like interface)."""
        self._ensure_loaded()
        return self._embeddings.shape[0]

    def search(self, query_embedding: np.ndarray, k: int):
        """
        Search for k nearest neighbors (FAISS-like interface).

        Args:
            query_embedding: Query vector of shape (1, dim) or (dim,)
            k: Number of results

        Returns:
            Tuple of (distances, indices) arrays matching FAISS format
        """
        # Use Milvus search
        distances, indices = self.milvus_client.search(
            pdf_slug=self.pdf_slug,
            query_embedding=query_embedding,
            top_k=k
        )

        # Return in FAISS format: (distances, indices) with shape (1, k)
        return distances.reshape(1, -1), indices.reshape(1, -1)

    def reconstruct_n(self, start: int, n: int, output: np.ndarray):
        """
        Reconstruct vectors (FAISS-like interface).

        Args:
            start: Starting index
            n: Number of vectors to reconstruct
            output: Output array to fill
        """
        self._ensure_loaded()
        output[:] = self._embeddings[start:start + n]


class StorageBackend:
    """Base class for storage backends."""

    def save_pdf_metadata(self, slug: str, **kwargs) -> None:
        raise NotImplementedError

    def get_pdf_metadata(self, slug: str) -> Optional[dict]:
        raise NotImplementedError

    def list_pdfs(self) -> list[dict]:
        raise NotImplementedError

    def save_chunks(self, slug: str, chunks: list[dict]) -> None:
        raise NotImplementedError

    def get_chunks(self, slug: str) -> list[dict]:
        raise NotImplementedError

    def save_embeddings(self, slug: str, embeddings: np.ndarray) -> None:
        raise NotImplementedError

    def load_embeddings(self, slug: str) -> np.ndarray:
        raise NotImplementedError

    def save_metadata(self, slug: str, key: str, data: Any) -> None:
        raise NotImplementedError

    def get_metadata(self, slug: str, key: str) -> Any:
        raise NotImplementedError

    def load_chunks(self, slug: str) -> list[dict]:
        """Alias for get_chunks for consistency."""
        return self.get_chunks(slug)

    def load_all_metadata(self, slug: str) -> dict:
        """Load all metadata for a PDF."""
        return {
            'sections': self.get_metadata(slug, 'sections') or {},
            'toc': self.get_metadata(slug, 'toc') or [],
            'mentions': self.get_metadata(slug, 'mentions') or [],
            'figures': self.get_metadata(slug, 'figures') or {},
            'tables': self.get_metadata(slug, 'tables') or {},
        }

    def get_toc(self, slug: str) -> list[dict]:
        """Fetch table of contents entries for a PDF."""
        return self.get_metadata(slug, 'toc') or []


class ArangoStorage(StorageBackend):
    """ArangoDB storage backend with Milvus for vector embeddings."""

    def __init__(self, base_dir: Path = Path("data"), use_milvus: bool = True):
        from pdfkg.db import ArangoDBClient, MilvusClient

        self.db_client = ArangoDBClient()
        # Note: connection happens in get_storage_backend()

        # Use Milvus for embeddings if available, otherwise fallback to FAISS
        self.use_milvus = use_milvus
        self.milvus_client = None

        if self.use_milvus:
            try:
                self.milvus_client = MilvusClient()
                print("✅ Milvus client initialized for vector storage")
            except Exception as e:
                print(f"⚠️  Milvus initialization failed: {e}")
                print("⚠️  Falling back to FAISS for embeddings")
                self.use_milvus = False

        # Fallback: Keep FAISS indexes in filesystem if Milvus is not available
        if not self.use_milvus:
            self.faiss_dir = base_dir / "faiss_indexes"
            self.faiss_dir.mkdir(parents=True, exist_ok=True)

    def save_pdf_metadata(self, slug: str, **kwargs) -> None:
        self.db_client.register_pdf(slug=slug, **kwargs)

    def get_pdf_metadata(self, slug: str) -> Optional[dict]:
        return self.db_client.get_pdf(slug)

    def list_pdfs(self) -> list[dict]:
        return self.db_client.list_pdfs()

    def save_chunks(self, slug: str, chunks: list[dict]) -> None:
        # Convert pandas dataframe if needed
        if isinstance(chunks, pd.DataFrame):
            chunks = chunks.to_dict("records")
        self.db_client.save_chunks(slug, chunks)

    def get_chunks(self, slug: str) -> list[dict]:
        return self.db_client.get_chunks(slug)

    def save_embeddings(self, slug: str, embeddings: np.ndarray, chunk_ids: list = None) -> None:
        """Save embeddings to Milvus or FAISS."""
        if self.use_milvus and self.milvus_client:
            # Save to Milvus
            try:
                self.milvus_client.save_embeddings(slug, embeddings, chunk_ids)
            except Exception as e:
                print(f"⚠️  Milvus save failed: {e}, falling back to FAISS")
                # Fallback to FAISS
                if not hasattr(self, 'faiss_dir'):
                    self.faiss_dir = Path("data") / "faiss_indexes"
                    self.faiss_dir.mkdir(parents=True, exist_ok=True)
                index = faiss.IndexFlatIP(embeddings.shape[1])
                index.add(embeddings)
                faiss.write_index(index, str(self.faiss_dir / f"{slug}.faiss"))
        else:
            # Save FAISS index to filesystem
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            faiss.write_index(index, str(self.faiss_dir / f"{slug}.faiss"))

    def load_embeddings(self, slug: str):
        """Load embeddings from Milvus or FAISS (returns Milvus-compatible interface)."""
        if self.use_milvus and self.milvus_client:
            # Load from Milvus - return a wrapper that provides FAISS-like interface
            try:
                return MilvusIndexWrapper(self.milvus_client, slug)
            except Exception as e:
                print(f"⚠️  Milvus load failed: {e}, falling back to FAISS")
                # Fallback to FAISS if available
                if hasattr(self, 'faiss_dir'):
                    faiss_path = self.faiss_dir / f"{slug}.faiss"
                    if faiss_path.exists():
                        return faiss.read_index(str(faiss_path))
                raise
        else:
            return faiss.read_index(str(self.faiss_dir / f"{slug}.faiss"))

    def save_metadata(self, slug: str, key: str, data: Any) -> None:
        self.db_client.save_metadata(slug, key, data)

    def get_metadata(self, slug: str, key: str) -> Any:
        return self.db_client.get_metadata(slug, key)

    def save_graph(self, slug: str, nodes: list[dict], edges: list[dict]) -> None:
        self.db_client.save_graph(slug, nodes, edges)

    def get_graph(self, slug: str) -> tuple[list[dict], list[dict]]:
        return self.db_client.get_graph(slug)

    def execute_aql(self, aql_query: str) -> list[dict]:
        """Executes a raw AQL query."""
        return self.db_client.execute_aql(aql_query)

    def get_chunks_by_keys(self, keys: list[str]) -> list[dict]:
        """Retrieves a list of chunk documents by their keys."""
        return self.db_client.get_chunks_by_keys(keys)

    def load_graph(self, slug: str):
        """Load graph from database and convert to NetworkX."""
        import networkx as nx
        nodes_data, edges_data = self.get_graph(slug)

        graph = nx.MultiDiGraph()

        # Add nodes
        for node in nodes_data:
            node_id = node.pop('node_id')
            graph.add_node(node_id, **node)

        # Add edges
        for edge in edges_data:
            from_id = edge.pop('from_id')
            to_id = edge.pop('to_id')
            graph.add_edge(from_id, to_id, **edge)

        return graph


class FileStorage(StorageBackend):
    """File-based storage backend (legacy)."""

    def __init__(self, base_dir: Path = Path("data")):
        from pdfkg.pdf_manager import PDFManager

        self.manager = PDFManager(base_dir)

    def save_pdf_metadata(self, slug: str, **kwargs) -> None:
        # Map kwargs to PDFManager.register_pdf signature
        self.manager.register_pdf(
            filename=kwargs.get("filename"),
            num_pages=kwargs.get("num_pages"),
            num_chunks=kwargs.get("num_chunks"),
            num_sections=kwargs.get("num_sections"),
            slug=slug,
            metadata=kwargs.get("metadata"),
        )

    def get_pdf_metadata(self, slug: str) -> Optional[dict]:
        return self.manager.get_pdf_info(slug)

    def list_pdfs(self) -> list[dict]:
        return self.manager.list_pdfs()

    def save_chunks(self, slug: str, chunks: list[dict]) -> None:
        out_dir = self.manager.get_pdf_output_dir(slug)
        if isinstance(chunks, pd.DataFrame):
            chunks.to_parquet(out_dir / "chunks.parquet", index=False)
        else:
            df = pd.DataFrame(chunks)
            df.to_parquet(out_dir / "chunks.parquet", index=False)

    def get_chunks(self, slug: str) -> list[dict]:
        out_dir = self.manager.get_pdf_output_dir(slug)
        df = pd.read_parquet(out_dir / "chunks.parquet")
        return df.to_dict("records")

    def save_embeddings(self, slug: str, embeddings: np.ndarray) -> None:
        out_dir = self.manager.get_pdf_output_dir(slug)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(out_dir / "index.faiss"))

    def load_embeddings(self, slug: str) -> faiss.Index:
        out_dir = self.manager.get_pdf_output_dir(slug)
        return faiss.read_index(str(out_dir / "index.faiss"))

    def save_metadata(self, slug: str, key: str, data: Any) -> None:
        out_dir = self.manager.get_pdf_output_dir(slug)
        (out_dir / f"{key}.json").write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    def get_metadata(self, slug: str, key: str) -> Any:
        out_dir = self.manager.get_pdf_output_dir(slug)
        path = out_dir / f"{key}.json"
        if path.exists():
            return orjson.loads(path.read_bytes())
        return None


def get_storage_backend() -> StorageBackend:
    """Get configured storage backend."""
    storage_type = os.getenv("STORAGE_BACKEND", "arango").lower()
    use_milvus = os.getenv("USE_MILVUS", "true").lower() in ("true", "1", "yes")

    if storage_type == "arango":
        try:
            storage = ArangoStorage(use_milvus=use_milvus)
            # Test ArangoDB connection
            storage.db_client.connect()

            # Test Milvus connection if enabled
            if use_milvus and storage.milvus_client:
                try:
                    storage.milvus_client.connect()
                except Exception as e:
                    print(f"⚠️  Milvus connection failed: {e}")
                    print(f"⚠️  Will use FAISS for embeddings instead")
                    storage.use_milvus = False

            return storage
        except Exception as e:
            print(f"⚠️  ArangoDB connection failed: {e}")
            print(f"⚠️  Falling back to file storage")
            print(f"   To use ArangoDB, run: ./start_arango.sh")
            return FileStorage()
    elif storage_type == "file":
        return FileStorage()
    else:
        raise ValueError(f"Unknown storage backend: {storage_type}")
