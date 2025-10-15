"""
Milvus vector database client for pdfkg embeddings.

This module provides a client for storing and querying embeddings using Milvus,
a purpose-built vector database optimized for similarity search.
"""

import os
from typing import List, Optional, Tuple
import numpy as np
from dotenv import load_dotenv

try:
    from pymilvus import (
        connections,
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
        utility,
    )
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False


load_dotenv()


class MilvusClient:
    """
    Client for interacting with Milvus vector database.

    Stores embeddings with associated chunk metadata for multi-PDF support.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: str = "pdfkg_embeddings",
    ):
        """
        Initialize Milvus client.

        Args:
            host: Milvus server host (default: from MILVUS_HOST env var or 'localhost')
            port: Milvus server port (default: from MILVUS_PORT env var or 19530)
            collection_name: Name of the collection to use
        """
        if not MILVUS_AVAILABLE:
            raise ImportError(
                "pymilvus is not installed. Install with: pip install pymilvus"
            )

        self.host = host or os.getenv("MILVUS_HOST", "localhost")
        self.port = int(port or os.getenv("MILVUS_PORT", "19530"))
        self.collection_name = collection_name
        self.collection: Optional[Collection] = None
        self._connected = False

    def connect(self) -> None:
        """Connect to Milvus server."""
        if self._connected:
            return

        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
            )
            self._connected = True
            print(f"✅ Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus: {e}")

    def disconnect(self) -> None:
        """Disconnect from Milvus server."""
        if self._connected:
            connections.disconnect(alias="default")
            self._connected = False
            self.collection = None

    def _create_collection(self, dimension: int) -> Collection:
        """
        Create the embeddings collection with schema.

        Args:
            dimension: Embedding vector dimension

        Returns:
            Created collection
        """
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="pdf_slug", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="PDF chunk embeddings for pdfkg"
        )

        # Create collection
        collection = Collection(
            name=self.collection_name,
            schema=schema,
            using="default",
        )

        # Create index for vector field
        index_params = {
            "metric_type": "IP",  # Inner Product (for cosine similarity with normalized vectors)
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }

        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )

        print(f"✅ Created Milvus collection '{self.collection_name}' with dimension {dimension}")

        return collection

    def _get_or_create_collection(self, dimension: Optional[int] = None) -> Collection:
        """
        Get existing collection or create new one.

        Args:
            dimension: Embedding dimension (required if creating new collection)

        Returns:
            Collection instance
        """
        if not self._connected:
            self.connect()

        # Check if collection exists
        if utility.has_collection(self.collection_name):
            collection = Collection(name=self.collection_name)
            collection.load()
            return collection
        else:
            if dimension is None:
                raise ValueError(
                    f"Collection '{self.collection_name}' does not exist and no dimension provided"
                )
            return self._create_collection(dimension)

    def reset_collection(self, dimension: Optional[int] = None, recreate: bool = True) -> None:
        """Drop pdfkg collection from Milvus and optionally recreate an empty one."""
        if not self._connected:
            self.connect()

        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print(f"🗑️  Dropped Milvus collection '{self.collection_name}'")

        self.collection = None

        if recreate:
            target_dim = dimension or int(os.getenv("DEFAULT_EMBED_DIM", "384"))
            self.collection = self._create_collection(target_dim)

    def save_embeddings(
        self,
        pdf_slug: str,
        embeddings: np.ndarray,
        chunk_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Save embeddings for a PDF to Milvus.

        Args:
            pdf_slug: PDF identifier
            embeddings: Numpy array of shape (n_chunks, embedding_dim)
            chunk_ids: Optional list of chunk IDs (will generate indices if not provided)
        """
        if not self._connected:
            self.connect()

        n_chunks, dimension = embeddings.shape

        # Delete existing embeddings for this PDF
        self.delete_embeddings(pdf_slug)

        # Get or create collection
        collection = self._get_or_create_collection(dimension)

        # Prepare data
        if chunk_ids is None:
            chunk_ids = [f"{pdf_slug}_chunk_{i}" for i in range(n_chunks)]

        data = [
            [pdf_slug] * n_chunks,  # pdf_slug
            chunk_ids,  # chunk_id
            list(range(n_chunks)),  # chunk_index
            embeddings.tolist(),  # embedding
        ]

        # Insert data
        collection.insert(data)
        collection.flush()

        print(f"✅ Saved {n_chunks} embeddings for '{pdf_slug}' to Milvus")

    def load_embeddings(
        self,
        pdf_slug: str,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Load embeddings for a PDF from Milvus.

        Args:
            pdf_slug: PDF identifier

        Returns:
            Tuple of (embeddings array, list of chunk IDs)
        """
        if not self._connected:
            self.connect()

        collection = self._get_or_create_collection()

        # Query all embeddings for this PDF
        results = collection.query(
            expr=f'pdf_slug == "{pdf_slug}"',
            output_fields=["chunk_id", "chunk_index", "embedding"],
        )

        if not results:
            raise ValueError(f"No embeddings found for PDF '{pdf_slug}'")

        # Sort by chunk_index to maintain order
        results = sorted(results, key=lambda x: x["chunk_index"])

        # Extract embeddings and chunk_ids
        embeddings = np.array([r["embedding"] for r in results], dtype=np.float32)
        chunk_ids = [r["chunk_id"] for r in results]

        print(f"✅ Loaded {len(embeddings)} embeddings for '{pdf_slug}' from Milvus")

        return embeddings, chunk_ids

    def search(
        self,
        pdf_slug: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar embeddings in a specific PDF.

        Args:
            pdf_slug: PDF identifier to search within
            query_embedding: Query vector of shape (1, embedding_dim) or (embedding_dim,)
            top_k: Number of results to return

        Returns:
            Tuple of (distances, indices) arrays
        """
        if not self._connected:
            self.connect()

        collection = self._get_or_create_collection()

        # Ensure query_embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search parameters
        search_params = {
            "metric_type": "IP",  # Inner Product
            "params": {"nprobe": 10},
        }

        # Perform search with filter
        results = collection.search(
            data=query_embedding.tolist(),
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=f'pdf_slug == "{pdf_slug}"',
            output_fields=["chunk_index"],
        )

        # Extract distances and indices
        if results and len(results) > 0:
            hits = results[0]  # First query result
            distances = np.array([hit.distance for hit in hits], dtype=np.float32)
            indices = np.array([hit.entity.get("chunk_index") for hit in hits], dtype=np.int64)
        else:
            # No results found
            distances = np.array([], dtype=np.float32)
            indices = np.array([], dtype=np.int64)

        return distances, indices

    def search_global(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> List[dict]:
        """
        Search for similar embeddings across ALL PDFs (global search).

        Args:
            query_embedding: Query vector of shape (1, embedding_dim) or (embedding_dim,)
            top_k: Number of results to return

        Returns:
            List of dicts with keys: pdf_slug, chunk_id, chunk_index, distance
        """
        if not self._connected:
            self.connect()

        collection = self._get_or_create_collection()

        # Ensure query_embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search parameters
        search_params = {
            "metric_type": "IP",  # Inner Product
            "params": {"nprobe": 10},
        }

        # Perform search WITHOUT pdf_slug filter (global search)
        results = collection.search(
            data=query_embedding.tolist(),
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["pdf_slug", "chunk_id", "chunk_index"],
        )

        # Extract results with metadata
        search_results = []
        if results and len(results) > 0:
            hits = results[0]  # First query result
            for hit in hits:
                search_results.append({
                    "pdf_slug": hit.entity.get("pdf_slug"),
                    "chunk_id": hit.entity.get("chunk_id"),
                    "chunk_index": hit.entity.get("chunk_index"),
                    "distance": float(hit.distance),
                })

        return search_results

    def delete_embeddings(self, pdf_slug: str) -> None:
        """
        Delete all embeddings for a PDF.

        Args:
            pdf_slug: PDF identifier
        """
        if not self._connected:
            self.connect()

        if not utility.has_collection(self.collection_name):
            return  # Collection doesn't exist, nothing to delete

        collection = Collection(name=self.collection_name)
        collection.load()  # Must load collection before deleting

        # Delete entities matching the pdf_slug
        collection.delete(expr=f'pdf_slug == "{pdf_slug}"')
        collection.flush()

        print(f"✅ Deleted embeddings for '{pdf_slug}' from Milvus")

    def list_pdfs(self) -> List[str]:
        """
        List all PDF slugs that have embeddings in Milvus.

        Returns:
            List of PDF slugs
        """
        if not self._connected:
            self.connect()

        if not utility.has_collection(self.collection_name):
            return []

        collection = self._get_or_create_collection()

        # Query distinct pdf_slugs
        results = collection.query(
            expr="pdf_slug != ''",
            output_fields=["pdf_slug"],
            limit=10000,  # Adjust as needed
        )

        # Get unique slugs
        slugs = list(set(r["pdf_slug"] for r in results))

        return sorted(slugs)

    def get_stats(self) -> dict:
        """
        Get statistics about the Milvus collection.

        Returns:
            Dict with collection statistics
        """
        if not self._connected:
            self.connect()

        if not utility.has_collection(self.collection_name):
            return {
                "collection_exists": False,
                "num_entities": 0,
            }

        collection = self._get_or_create_collection()

        stats = {
            "collection_exists": True,
            "num_entities": collection.num_entities,
            "collection_name": self.collection_name,
        }

        return stats
