"""
ArangoDB client for multi-PDF knowledge graph platform.
"""

import os
from datetime import datetime
from typing import Any, Optional

from arango import ArangoClient
from arango.database import StandardDatabase
from dotenv import load_dotenv

load_dotenv()


class ArangoDBClient:
    """ArangoDB client for storing PDFs, chunks, and knowledge graphs."""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        username: str = None,
        password: str = None,
        db_name: str = None,
    ):
        """
        Initialize ArangoDB client.

        Args:
            host: ArangoDB host (default: from env or localhost)
            port: ArangoDB port (default: from env or 8529)
            username: Username (default: from env or root)
            password: Password (default: from env or empty)
            db_name: Database name (default: from env or pdfkg)
        """
        self.host = host or os.getenv("ARANGO_HOST", "localhost")
        self.port = int(port or os.getenv("ARANGO_PORT", "8529"))
        self.username = username or os.getenv("ARANGO_USER", "root")
        self.password = password or os.getenv("ARANGO_PASSWORD", "")
        self.db_name = db_name or os.getenv("ARANGO_DB", "pdfkg")

        # Initialize client
        self.client = ArangoClient(hosts=f"http://{self.host}:{self.port}")
        self.db: Optional[StandardDatabase] = None

        # Collection names
        self.PDFS = "pdfs"
        self.NODES = "nodes"
        self.EDGES = "edges"
        self.CHUNKS = "chunks"
        self.QA_HISTORY = "qa_history"

        # Cross-document relationship collections
        self.ENTITIES = "entities"
        self.ENTITY_MENTIONS = "entity_mentions"
        self.SEMANTIC_LINKS = "semantic_links"
        self.CROSS_DOC_REFS = "cross_doc_refs"
        self.VERSION_RELATIONS = "version_relations"
        self.TOPICS = "topics"
        self.TOPIC_ASSIGNMENTS = "topic_assignments"

    def connect(self) -> StandardDatabase:
        """Connect to ArangoDB and create database if needed."""
        # Connect to _system database first
        sys_db = self.client.db("_system", username=self.username, password=self.password)

        # Create database if it doesn't exist
        if not sys_db.has_database(self.db_name):
            sys_db.create_database(self.db_name)
            print(f"Created database: {self.db_name}")

        # Connect to target database
        self.db = self.client.db(self.db_name, username=self.username, password=self.password)

        # Initialize collections
        self._init_collections()

        return self.db

    def _init_collections(self) -> None:
        """Initialize collections and indexes."""
        # PDFs collection (document)
        if not self.db.has_collection(self.PDFS):
            self.db.create_collection(self.PDFS)
            # Index on slug for fast lookups
            self.db.collection(self.PDFS).add_hash_index(fields=["slug"], unique=True)
            print(f"Created collection: {self.PDFS}")

        # Nodes collection (document) - pages, sections, paragraphs, figures, tables
        if not self.db.has_collection(self.NODES):
            self.db.create_collection(self.NODES)
            # Indexes
            self.db.collection(self.NODES).add_hash_index(fields=["pdf_slug"])
            self.db.collection(self.NODES).add_hash_index(fields=["type"])
            self.db.collection(self.NODES).add_hash_index(fields=["pdf_slug", "type"])
            print(f"Created collection: {self.NODES}")

        # Edges collection (edge) - relationships between nodes
        if not self.db.has_collection(self.EDGES):
            self.db.create_collection(self.EDGES, edge=True)
            # Indexes
            self.db.collection(self.EDGES).add_hash_index(fields=["pdf_slug"])
            self.db.collection(self.EDGES).add_hash_index(fields=["type"])
            print(f"Created collection: {self.EDGES}")

        # Chunks collection (document) - optimized for text search
        if not self.db.has_collection(self.CHUNKS):
            self.db.create_collection(self.CHUNKS)
            # Indexes
            self.db.collection(self.CHUNKS).add_hash_index(fields=["pdf_slug"])
            self.db.collection(self.CHUNKS).add_hash_index(fields=["section_id"])
            self.db.collection(self.CHUNKS).add_hash_index(fields=["chunk_id"], unique=True)
            # Full-text index on text
            self.db.collection(self.CHUNKS).add_fulltext_index(fields=["text"])
            print(f"Created collection: {self.CHUNKS}")

        # Q&A History collection (document) - for auditing and debugging
        if not self.db.has_collection(self.QA_HISTORY):
            self.db.create_collection(self.QA_HISTORY)
            # Indexes
            self.db.collection(self.QA_HISTORY).add_hash_index(fields=["pdf_slug"])
            self.db.collection(self.QA_HISTORY).add_hash_index(fields=["llm_provider"])
            self.db.collection(self.QA_HISTORY).add_persistent_index(fields=["timestamp"])
            # Full-text index on question
            self.db.collection(self.QA_HISTORY).add_fulltext_index(fields=["question"])
            print(f"Created collection: {self.QA_HISTORY}")

        # === Cross-document relationship collections ===

        # Entities collection (document) - shared entities across PDFs
        if not self.db.has_collection(self.ENTITIES):
            self.db.create_collection(self.ENTITIES)
            # Indexes
            self.db.collection(self.ENTITIES).add_hash_index(fields=["canonical_name"], unique=False)
            self.db.collection(self.ENTITIES).add_hash_index(fields=["entity_type"])
            self.db.collection(self.ENTITIES).add_fulltext_index(fields=["canonical_name"])
            print(f"Created collection: {self.ENTITIES}")

        # Entity mentions collection (edge) - chunk -> entity
        if not self.db.has_collection(self.ENTITY_MENTIONS):
            self.db.create_collection(self.ENTITY_MENTIONS, edge=True)
            # Indexes
            self.db.collection(self.ENTITY_MENTIONS).add_hash_index(fields=["pdf_slug"])
            print(f"Created collection: {self.ENTITY_MENTIONS}")

        # Semantic links collection (edge) - chunk -> chunk semantic similarity
        if not self.db.has_collection(self.SEMANTIC_LINKS):
            self.db.create_collection(self.SEMANTIC_LINKS, edge=True)
            # Indexes
            self.db.collection(self.SEMANTIC_LINKS).add_hash_index(fields=["source_pdf"])
            self.db.collection(self.SEMANTIC_LINKS).add_hash_index(fields=["target_pdf"])
            self.db.collection(self.SEMANTIC_LINKS).add_persistent_index(fields=["similarity"])
            print(f"Created collection: {self.SEMANTIC_LINKS}")

        # Cross-document references collection (edge) - chunk -> pdf
        if not self.db.has_collection(self.CROSS_DOC_REFS):
            self.db.create_collection(self.CROSS_DOC_REFS, edge=True)
            # Indexes
            self.db.collection(self.CROSS_DOC_REFS).add_hash_index(fields=["source_pdf"])
            self.db.collection(self.CROSS_DOC_REFS).add_hash_index(fields=["target_pdf"])
            print(f"Created collection: {self.CROSS_DOC_REFS}")

        # Version relations collection (edge) - pdf -> pdf version relationship
        if not self.db.has_collection(self.VERSION_RELATIONS):
            self.db.create_collection(self.VERSION_RELATIONS, edge=True)
            # Indexes
            self.db.collection(self.VERSION_RELATIONS).add_hash_index(fields=["relationship_type"])
            print(f"Created collection: {self.VERSION_RELATIONS}")

        # Topics collection (document) - document clusters/topics
        if not self.db.has_collection(self.TOPICS):
            self.db.create_collection(self.TOPICS)
            # Indexes
            self.db.collection(self.TOPICS).add_hash_index(fields=["topic_id"], unique=True)
            print(f"Created collection: {self.TOPICS}")

        # Topic assignments collection (edge) - pdf -> topic
        if not self.db.has_collection(self.TOPIC_ASSIGNMENTS):
            self.db.create_collection(self.TOPIC_ASSIGNMENTS, edge=True)
            # Indexes
            self.db.collection(self.TOPIC_ASSIGNMENTS).add_hash_index(fields=["topic_id"])
            self.db.collection(self.TOPIC_ASSIGNMENTS).add_persistent_index(fields=["probability"])
            print(f"Created collection: {self.TOPIC_ASSIGNMENTS}")

    def reset_database(self) -> None:
        """Drop and recreate the pdfkg database and its collections."""
        sys_db = self.client.db("_system", username=self.username, password=self.password)

        if sys_db.has_database(self.db_name):
            sys_db.delete_database(self.db_name)
            print(f"🗑️  Dropped database: {self.db_name}")

        # Reset local handle and recreate schema
        self.db = None
        self.connect()

    def register_pdf(
        self,
        slug: str,
        filename: str,
        num_pages: int,
        num_chunks: int,
        num_sections: int,
        num_figures: int = 0,
        num_tables: int = 0,
        metadata: dict = None,
    ) -> dict:
        """
        Register a processed PDF.

        Args:
            slug: PDF slug (unique identifier)
            filename: Original filename
            num_pages: Number of pages
            num_chunks: Number of text chunks
            num_sections: Number of sections
            num_figures: Number of figures
            num_tables: Number of tables
            metadata: Additional metadata

        Returns:
            PDF document
        """
        pdf_doc = {
            "_key": slug,
            "slug": slug,
            "filename": filename,
            "processed_date": datetime.now().isoformat(),
            "num_pages": num_pages,
            "num_chunks": num_chunks,
            "num_sections": num_sections,
            "num_figures": num_figures,
            "num_tables": num_tables,
            "metadata": metadata or {},
        }

        # Insert or update
        pdfs = self.db.collection(self.PDFS)
        if pdfs.has(slug):
            pdfs.update(pdf_doc)
        else:
            pdfs.insert(pdf_doc)

        return pdf_doc

    def get_pdf(self, slug: str) -> Optional[dict]:
        """Get PDF metadata by slug."""
        try:
            return self.db.collection(self.PDFS).get(slug)
        except:
            return None

    def list_pdfs(self) -> list[dict]:
        """List all registered PDFs."""
        cursor = self.db.aql.execute(
            """
            FOR pdf IN @@collection
            SORT pdf.processed_date DESC
            RETURN pdf
            """,
            bind_vars={"@collection": self.PDFS},
        )
        return list(cursor)

    def pdf_exists(self, slug: str) -> bool:
        """Check if PDF exists."""
        return self.db.collection(self.PDFS).has(slug)

    def delete_pdf(self, slug: str) -> bool:
        """
        Delete a PDF and all its associated data.

        Args:
            slug: PDF slug

        Returns:
            True if deleted, False if not found
        """
        if not self.pdf_exists(slug):
            return False

        # Delete chunks
        self.db.aql.execute(
            """
            FOR chunk IN @@collection
            FILTER chunk.pdf_slug == @slug
            REMOVE chunk IN @@collection
            """,
            bind_vars={"@collection": self.CHUNKS, "slug": slug},
        )

        # Delete nodes
        self.db.aql.execute(
            """
            FOR node IN @@collection
            FILTER node.pdf_slug == @slug
            REMOVE node IN @@collection
            """,
            bind_vars={"@collection": self.NODES, "slug": slug},
        )

        # Delete edges
        self.db.aql.execute(
            """
            FOR edge IN @@collection
            FILTER edge.pdf_slug == @slug
            REMOVE edge IN @@collection
            """,
            bind_vars={"@collection": self.EDGES, "slug": slug},
        )

        # Delete PDF
        self.db.collection(self.PDFS).delete(slug)

        return True

    def save_chunks(self, pdf_slug: str, chunks: list[dict]) -> None:
        """
        Save text chunks for a PDF.

        Args:
            pdf_slug: PDF slug
            chunks: List of chunk dicts with keys: chunk_id, section_id, page, text
        """
        chunks_collection = self.db.collection(self.CHUNKS)

        # Add pdf_slug to each chunk
        for chunk in chunks:
            chunk["pdf_slug"] = pdf_slug
            chunk["_key"] = chunk["chunk_id"]

        # Batch insert
        chunks_collection.insert_many(chunks, overwrite=True)

    def get_chunks(self, pdf_slug: str, limit: int = None) -> list[dict]:
        """
        Get all chunks for a PDF.

        Args:
            pdf_slug: PDF slug
            limit: Maximum number of chunks to return

        Returns:
            List of chunk documents
        """
        query = """
            FOR chunk IN @@collection
            FILTER chunk.pdf_slug == @slug
            SORT chunk.page, chunk.section_id
            LIMIT @limit
            RETURN chunk
        """
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                "@collection": self.CHUNKS,
                "slug": pdf_slug,
                "limit": limit or 999999,
            },
        )
        return list(cursor)

    def save_graph(self, pdf_slug: str, nodes: list[dict], edges: list[dict]) -> None:
        """
        Save knowledge graph nodes and edges.

        Args:
            pdf_slug: PDF slug
            nodes: List of node dicts with keys: node_id, type, label, ...
            edges: List of edge dicts with keys: _from, _to, type, ...
        """
        nodes_collection = self.db.collection(self.NODES)
        edges_collection = self.db.collection(self.EDGES)

        # Add pdf_slug and prepare keys
        for node in nodes:
            node["pdf_slug"] = pdf_slug
            node["_key"] = node["node_id"].replace(":", "_").replace("/", "_")

        for edge in edges:
            edge["pdf_slug"] = pdf_slug
            # ArangoDB edges need _from and _to in format "collection/key"
            edge["_from"] = f"{self.NODES}/{edge['from_id'].replace(':', '_').replace('/', '_')}"
            edge["_to"] = f"{self.NODES}/{edge['to_id'].replace(':', '_').replace('/', '_')}"

        # Batch insert
        nodes_collection.insert_many(nodes, overwrite=True)
        edges_collection.insert_many(edges, overwrite=True)

        # Create or update named graph for visualization
        self._ensure_named_graph(pdf_slug)

    def get_graph(self, pdf_slug: str) -> tuple[list[dict], list[dict]]:
        """
        Get knowledge graph for a PDF.

        Args:
            pdf_slug: PDF slug

        Returns:
            Tuple of (nodes, edges)
        """
        # Get nodes
        nodes_cursor = self.db.aql.execute(
            """
            FOR node IN @@collection
            FILTER node.pdf_slug == @slug
            RETURN node
            """,
            bind_vars={"@collection": self.NODES, "slug": pdf_slug},
        )
        nodes = list(nodes_cursor)

        # Get edges
        edges_cursor = self.db.aql.execute(
            """
            FOR edge IN @@collection
            FILTER edge.pdf_slug == @slug
            RETURN edge
            """,
            bind_vars={"@collection": self.EDGES, "slug": pdf_slug},
        )
        edges = list(edges_cursor)

        return nodes, edges

    def save_metadata(self, pdf_slug: str, key: str, data: Any) -> None:
        """
        Save arbitrary metadata for a PDF.

        Args:
            pdf_slug: PDF slug (use '__global__' for cross-document metadata)
            key: Metadata key (e.g., 'sections', 'toc', 'mentions')
            data: Data to save
        """
        # Handle special __global__ key for cross-document metadata
        if pdf_slug == "__global__":
            pdfs_collection = self.db.collection(self.PDFS)
            # Check if __global__ document exists
            if not pdfs_collection.has("__global__"):
                # Create it
                pdfs_collection.insert({
                    "_key": "__global__",
                    "slug": "__global__",
                    "filename": "__global__",
                    "processed_date": datetime.now().isoformat(),
                    "num_pages": 0,
                    "num_chunks": 0,
                    "num_sections": 0,
                    "num_figures": 0,
                    "num_tables": 0,
                    "metadata": {}
                })

            # Get the document
            pdf = pdfs_collection.get("__global__")
            pdf["metadata"][key] = data
            pdfs_collection.update({"_key": "__global__", "metadata": pdf["metadata"]})
            return

        # Regular PDF metadata
        pdf = self.get_pdf(pdf_slug)
        if not pdf:
            raise ValueError(f"PDF not found: {pdf_slug}")

        pdf["metadata"][key] = data
        self.db.collection(self.PDFS).update({"_key": pdf_slug, "metadata": pdf["metadata"]})

    def get_metadata(self, pdf_slug: str, key: str) -> Any:
        """
        Get metadata for a PDF.

        Args:
            pdf_slug: PDF slug (use '__global__' for cross-document metadata)
            key: Metadata key

        Returns:
            Metadata value or None
        """
        # Handle special __global__ key for cross-document metadata
        if pdf_slug == "__global__":
            pdfs_collection = self.db.collection(self.PDFS)
            if not pdfs_collection.has("__global__"):
                return None
            pdf = pdfs_collection.get("__global__")
            return pdf.get("metadata", {}).get(key)

        # Regular PDF metadata
        pdf = self.get_pdf(pdf_slug)
        if not pdf:
            return None
        return pdf.get("metadata", {}).get(key)

    def search_chunks_fulltext(self, pdf_slug: str, query: str, limit: int = 10) -> list[dict]:
        """
        Full-text search in chunks.

        Args:
            pdf_slug: PDF slug (None for all PDFs)
            query: Search query
            limit: Maximum results

        Returns:
            List of matching chunks
        """
        if pdf_slug:
            aql = """
                FOR chunk IN FULLTEXT(@@collection, 'text', @query)
                FILTER chunk.pdf_slug == @slug
                LIMIT @limit
                RETURN chunk
            """
            bind_vars = {"@collection": self.CHUNKS, "query": query, "slug": pdf_slug, "limit": limit}
        else:
            aql = """
                FOR chunk IN FULLTEXT(@@collection, 'text', @query)
                LIMIT @limit
                RETURN chunk
            """
            bind_vars = {"@collection": self.CHUNKS, "query": query, "limit": limit}

        cursor = self.db.aql.execute(aql, bind_vars=bind_vars)
        return list(cursor)

    def get_related_nodes(self, node_id: str, edge_types: list[str] = None, direction: str = "outbound") -> list[dict]:
        """
        Get nodes related to a given node via edges.

        Args:
            node_id: Source node ID
            edge_types: List of edge types to follow (None for all)
            direction: 'outbound', 'inbound', or 'any'

        Returns:
            List of related nodes
        """
        node_key = node_id.replace(":", "_").replace("/", "_")

        if edge_types:
            filter_clause = f"FILTER edge.type IN {edge_types}"
        else:
            filter_clause = ""

        aql = f"""
            FOR vertex, edge IN 1..1 {direction.upper()} '{self.NODES}/{node_key}' {self.EDGES}
            {filter_clause}
            RETURN vertex
        """

        cursor = self.db.aql.execute(aql)
        return list(cursor)

    def save_qa_interaction(
        self,
        question: str,
        answer: str,
        pdf_slug: str,
        llm_provider: str,
        llm_model: str,
        embed_model: str,
        top_k: int,
        sources: list[dict],
        related_items: dict,
        response_time_ms: float,
        timestamp: str = None,
    ) -> dict:
        """
        Save a Q&A interaction to the history collection for auditing.

        Args:
            question: User question
            answer: Generated answer
            pdf_slug: PDF slug
            llm_provider: LLM provider ("gemini", "mistral", or "none")
            llm_model: Specific model used
            embed_model: Embedding model used
            top_k: Number of chunks retrieved
            sources: Retrieved chunks with scores
            related_items: Related figures/tables/sections
            response_time_ms: Response time in milliseconds
            timestamp: ISO timestamp (auto-generated if None)

        Returns:
            Saved Q&A document
        """
        from datetime import datetime

        qa_doc = {
            "timestamp": timestamp or datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "pdf_slug": pdf_slug,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "embed_model": embed_model,
            "top_k": top_k,
            "sources": sources,
            "related_items": related_items,
            "response_time_ms": response_time_ms,
        }

        qa_collection = self.db.collection(self.QA_HISTORY)
        result = qa_collection.insert(qa_doc)
        qa_doc["_key"] = result["_key"]
        return qa_doc

    def get_qa_history(
        self, pdf_slug: str = None, limit: int = 100, llm_provider: str = None
    ) -> list[dict]:
        """
        Get Q&A history with optional filters.

        Args:
            pdf_slug: Filter by PDF slug (None for all)
            limit: Maximum results
            llm_provider: Filter by LLM provider (None for all)

        Returns:
            List of Q&A documents sorted by timestamp descending
        """
        filters = []
        bind_vars = {"@collection": self.QA_HISTORY, "limit": limit}

        if pdf_slug:
            filters.append("FILTER qa.pdf_slug == @pdf_slug")
            bind_vars["pdf_slug"] = pdf_slug

        if llm_provider:
            filters.append("FILTER qa.llm_provider == @llm_provider")
            bind_vars["llm_provider"] = llm_provider

        filter_clause = "\n".join(filters) if filters else ""

        query = f"""
            FOR qa IN @@collection
            {filter_clause}
            SORT qa.timestamp DESC
            LIMIT @limit
            RETURN qa
        """

        cursor = self.db.aql.execute(query, bind_vars=bind_vars)
        return list(cursor)

    def _ensure_named_graph(self, pdf_slug: str) -> None:
        """
        Create or update a named graph for a PDF to enable visualization in ArangoDB UI.

        Args:
            pdf_slug: PDF slug
        """
        graph_name = f"graph_{pdf_slug}"

        # Check if graph already exists
        if self.db.has_graph(graph_name):
            # Graph exists, just return
            return

        # Create named graph
        # Note: This creates a graph that includes ALL edges in the edges collection
        # that connect nodes in the nodes collection. We can filter by pdf_slug in queries.
        try:
            self.db.create_graph(
                graph_name,
                edge_definitions=[
                    {
                        "edge_collection": self.EDGES,
                        "from_vertex_collections": [self.NODES],
                        "to_vertex_collections": [self.NODES],
                    }
                ],
            )
            print(f"Created named graph: {graph_name}")
        except Exception as e:
            # Graph might already exist or other error
            print(f"Note: Could not create named graph {graph_name}: {e}")

    def execute_aql(self, aql_query: str) -> list[dict]:
        """
        Executes a raw AQL query.

        Args:
            aql_query: The AQL query string to execute.

        Returns:
            A list of result documents.
        """
        cursor = self.db.aql.execute(aql_query)
        return list(cursor)

    def get_chunks_by_keys(self, keys: list[str]) -> list[dict]:
        """
        Retrieves a list of chunk documents by their keys (_key).

        Args:
            keys: A list of document keys.

        Returns:
            A list of chunk documents.
        """
        if not keys:
            return []

        query = """
            FOR chunk IN @@collection
            FILTER chunk._key IN @keys
            RETURN chunk
        """
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                "@collection": self.CHUNKS,
                "keys": keys,
            },
        )
        return list(cursor)
