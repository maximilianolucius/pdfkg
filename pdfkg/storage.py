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


class ArangoStorage(StorageBackend):
    """ArangoDB storage backend."""

    def __init__(self, base_dir: Path = Path("data")):
        from pdfkg.db import ArangoDBClient

        self.db_client = ArangoDBClient()
        # Note: connection happens in get_storage_backend()
        # Keep FAISS indexes in filesystem for now
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

    def save_embeddings(self, slug: str, embeddings: np.ndarray) -> None:
        # Save FAISS index to filesystem
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(self.faiss_dir / f"{slug}.faiss"))

    def load_embeddings(self, slug: str) -> faiss.Index:
        return faiss.read_index(str(self.faiss_dir / f"{slug}.faiss"))

    def save_metadata(self, slug: str, key: str, data: Any) -> None:
        self.db_client.save_metadata(slug, key, data)

    def get_metadata(self, slug: str, key: str) -> Any:
        return self.db_client.get_metadata(slug, key)

    def save_graph(self, slug: str, nodes: list[dict], edges: list[dict]) -> None:
        self.db_client.save_graph(slug, nodes, edges)

    def get_graph(self, slug: str) -> tuple[list[dict], list[dict]]:
        return self.db_client.get_graph(slug)


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

    if storage_type == "arango":
        try:
            storage = ArangoStorage()
            # Test connection
            storage.db_client.connect()
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
