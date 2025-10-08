"""
Embedding and FAISS indexing utilities.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

from pdfkg.config import Chunk

# Model cache to prevent repeated loading (fixes macOS semaphore leak)
_model_cache = {}


def _get_model(model_name: str) -> SentenceTransformer:
    """Get or create cached model instance."""
    if model_name not in _model_cache:
        # Disable tokenizer parallelism to avoid fork issues on macOS
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        _model_cache[model_name] = SentenceTransformer(model_name, device="cpu")
    return _model_cache[model_name]


def embed_chunks(
    chunks: list[Chunk], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Embed chunks using sentence-transformers and normalize.

    Args:
        chunks: List of Chunk objects.
        model_name: Sentence-transformers model name.

    Returns:
        Numpy array of shape (n_chunks, dim) with normalized embeddings.
    """
    model = _get_model(model_name)
    texts = [c.text for c in chunks]
    # Use batch_size=32 and disable multi-process pool to avoid macOS fork issues
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=32
    )
    # Normalize for cosine similarity via inner product
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings.astype(np.float32)


def build_faiss_index(X: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build FAISS index using inner product (for normalized vectors = cosine).

    Args:
        X: Normalized embeddings of shape (n, dim).

    Returns:
        FAISS IndexFlatIP.
    """
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    return index


def encode_query(text: str, model_name: str) -> np.ndarray:
    """
    Encode a query text and normalize.

    Args:
        text: Query text.
        model_name: Sentence-transformers model name.

    Returns:
        Normalized embedding vector.
    """
    model = _get_model(model_name)
    embedding = model.encode([text], convert_to_numpy=True)
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding.astype(np.float32)
