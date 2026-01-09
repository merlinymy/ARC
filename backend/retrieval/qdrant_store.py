"""Qdrant vector store operations with hybrid search support."""

import logging
import uuid
from typing import List, Dict, Any, Optional, Set

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Prefetch,
    FusionQuery,
    Fusion,
    SparseVector as QdrantSparseVector,
    PayloadSchemaType,
)

from .bm25 import BM25Vectorizer, SparseVector, HybridSearchMixer

logger = logging.getLogger(__name__)


class QdrantStore:
    """Qdrant vector store for research paper chunks.

    Single collection with all 6 chunk types, differentiated by metadata.

    Supports dependency injection for shared clients to avoid multiple connections.
    """

    # Namespace UUID for deterministic ID generation (RFC 4122)
    NAMESPACE_UUID = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "research_papers",
        embedding_dimension: int = 1024,
        enable_hybrid: bool = False,
        client: Optional[QdrantClient] = None,
        bm25_vectorizer: Optional[BM25Vectorizer] = None,
    ):
        """Initialize QdrantStore.

        Args:
            host: Qdrant host (ignored if client provided)
            port: Qdrant port (ignored if client provided)
            collection_name: Name of the Qdrant collection
            embedding_dimension: Dimension of embeddings
            enable_hybrid: Enable hybrid search with BM25
            client: Optional shared QdrantClient (for dependency injection)
            bm25_vectorizer: Optional shared BM25Vectorizer (for dependency injection)
        """
        # Use injected client or create a new one
        if client is not None:
            self.client = client
            self._owns_client = False
            logger.info(f"Using injected Qdrant client for collection '{collection_name}'")
        else:
            self.client = QdrantClient(host=host, port=port)
            self._owns_client = True
            logger.info(f"Created Qdrant client at {host}:{port}")

        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.enable_hybrid = enable_hybrid or (bm25_vectorizer is not None)

        # Track indexed chunk IDs for cache invalidation
        self._recently_upserted_papers: Set[str] = set()

        # BM25 for hybrid search - use injected vectorizer or create new one
        if self.enable_hybrid:
            if bm25_vectorizer is not None:
                self.bm25_vectorizer = bm25_vectorizer
                logger.info("Using injected BM25 vectorizer")
            else:
                self.bm25_vectorizer = BM25Vectorizer()
                if self.bm25_vectorizer.load_idf_cache():
                    logger.info(f"Loaded BM25 IDF cache with {len(self.bm25_vectorizer._idf_cache)} terms")
            self.hybrid_mixer = HybridSearchMixer(dense_weight=0.7, sparse_weight=0.3)
        else:
            self.bm25_vectorizer = None
            self.hybrid_mixer = None

        logger.info(f"QdrantStore ready for collection '{collection_name}' (hybrid={self.enable_hybrid})")

    def _chunk_id_to_point_id(self, chunk_id: str) -> str:
        """Convert chunk_id to a deterministic UUID string for Qdrant.

        Uses UUID5 (SHA-1 based) which is deterministic and collision-resistant.
        Returns a UUID string that Qdrant accepts as a point ID.
        """
        return str(uuid.uuid5(self.NAMESPACE_UUID, chunk_id))

    def ensure_collection(self) -> bool:
        """Create collection if it doesn't exist, with payload indices."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        created = False
        if self.collection_name not in collection_names:
            # Create collection with sparse vector support for true hybrid search
            vectors_config = VectorParams(
                size=self.embedding_dimension,
                distance=Distance.COSINE
            )

            # Configure sparse vectors for BM25 hybrid search
            sparse_config = None
            if self.enable_hybrid:
                sparse_config = {
                    "bm25": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                }

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_config,
            )
            logger.info(f"Created collection: {self.collection_name}")
            created = True
        else:
            logger.info(f"Collection {self.collection_name} already exists")

        # Ensure payload indices exist for efficient filtering
        self._ensure_payload_indices()

        return created

    def _ensure_payload_indices(self) -> None:
        """Create payload indices for efficient filtering and lookups."""
        indices_to_create = [
            ("_chunk_id", PayloadSchemaType.KEYWORD),
            ("chunk_type", PayloadSchemaType.KEYWORD),
            ("paper_id", PayloadSchemaType.KEYWORD),
            ("section_name", PayloadSchemaType.KEYWORD),
        ]

        for field_name, schema_type in indices_to_create:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
                logger.debug(f"Created payload index: {field_name}")
            except Exception as e:
                # Index may already exist
                if "already exists" not in str(e).lower():
                    logger.warning(f"Failed to create index {field_name}: {e}")

    def upsert_chunks(
        self,
        chunk_ids: List[str],
        embeddings: List[List[float]],
        payloads: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """Upsert chunks to collection with optional sparse vectors for hybrid search."""
        total = 0

        # Track paper IDs for cache invalidation
        paper_ids_in_batch = set()

        for i in range(0, len(chunk_ids), batch_size):
            batch_ids = chunk_ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_payloads = payloads[i:i + batch_size]

            points = []
            for chunk_id, embedding, payload in zip(batch_ids, batch_embeddings, batch_payloads):
                # Use UUID5 for collision-resistant deterministic IDs
                point_id = self._chunk_id_to_point_id(chunk_id)

                # Track paper ID for cache invalidation
                if 'paper_id' in payload:
                    paper_ids_in_batch.add(payload['paper_id'])

                # Build vector dict (dense + optional sparse)
                vector_data: Any = embedding
                if self.enable_hybrid and self.bm25_vectorizer:
                    text = payload.get('text', '')
                    sparse_vec = self.bm25_vectorizer.vectorize(text, is_query=False)
                    vector_data = {
                        "": embedding,  # Default dense vector
                        "bm25": QdrantSparseVector(
                            indices=sparse_vec.indices,
                            values=sparse_vec.values,
                        )
                    }

                points.append(PointStruct(
                    id=point_id,
                    vector=vector_data,
                    payload={**payload, '_chunk_id': chunk_id}
                ))

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            total += len(points)

        # Track recently upserted papers for cache invalidation
        self._recently_upserted_papers.update(paper_ids_in_batch)

        logger.info(f"Upserted {total} chunks to {self.collection_name}")
        return total

    def get_recently_upserted_papers(self) -> Set[str]:
        """Get and clear the set of recently upserted paper IDs for cache invalidation."""
        papers = self._recently_upserted_papers.copy()
        self._recently_upserted_papers.clear()
        return papers

    def search(
        self,
        query_embedding: List[float],
        limit: int = 50,
        chunk_types: Optional[List[str]] = None,
        section_names: Optional[List[str]] = None,
        paper_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks with filtering."""
        filter_conditions = []

        if chunk_types:
            if len(chunk_types) == 1:
                filter_conditions.append(
                    FieldCondition(key="chunk_type", match=MatchValue(value=chunk_types[0]))
                )
            else:
                filter_conditions.append(
                    FieldCondition(key="chunk_type", match=MatchAny(any=chunk_types))
                )

        if section_names:
            if len(section_names) == 1:
                filter_conditions.append(
                    FieldCondition(key="section_name", match=MatchValue(value=section_names[0]))
                )
            else:
                filter_conditions.append(
                    FieldCondition(key="section_name", match=MatchAny(any=section_names))
                )

        if paper_ids:
            if len(paper_ids) == 1:
                filter_conditions.append(
                    FieldCondition(key="paper_id", match=MatchValue(value=paper_ids[0]))
                )
            else:
                filter_conditions.append(
                    FieldCondition(key="paper_id", match=MatchAny(any=paper_ids))
                )

        query_filter = Filter(must=filter_conditions) if filter_conditions else None

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
            query_filter=query_filter,
        )

        return [{'score': r.score, **r.payload} for r in results.points]

    def search_by_strategy(
        self,
        query_embedding: List[float],
        chunk_types: List[str],
        top_k: int = 50,
        section_filter: Optional[List[str]] = None,
        paper_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search using a retrieval strategy with a single optimized query."""
        # Use single query with MatchAny filter instead of multiple queries
        return self.search(
            query_embedding=query_embedding,
            limit=top_k,
            chunk_types=chunk_types,
            section_names=section_filter,
            paper_ids=paper_ids,
        )

    def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        limit: int = 50,
        chunk_types: Optional[List[str]] = None,
        section_names: Optional[List[str]] = None,
        paper_ids: Optional[List[str]] = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Perform true hybrid search combining dense and sparse (BM25) vectors.

        Uses Qdrant's native Prefetch + RRF fusion for optimal hybrid search.

        Args:
            query: Original query text for BM25
            query_embedding: Dense embedding of query
            limit: Maximum results to return
            chunk_types: Filter by chunk types
            section_names: Filter by section names
            paper_ids: Filter by paper IDs
            dense_weight: Weight for dense search scores (used in RRF)
            sparse_weight: Weight for sparse search scores (used in RRF)

        Returns:
            List of results with hybrid scores
        """
        if not self.enable_hybrid or not self.bm25_vectorizer:
            # Fall back to dense-only search
            return self.search(
                query_embedding=query_embedding,
                limit=limit,
                chunk_types=chunk_types,
                section_names=section_names,
                paper_ids=paper_ids,
            )

        # Build filter conditions
        filter_conditions = []
        if chunk_types:
            if len(chunk_types) == 1:
                filter_conditions.append(
                    FieldCondition(key="chunk_type", match=MatchValue(value=chunk_types[0]))
                )
            else:
                filter_conditions.append(
                    FieldCondition(key="chunk_type", match=MatchAny(any=chunk_types))
                )

        if section_names:
            if len(section_names) == 1:
                filter_conditions.append(
                    FieldCondition(key="section_name", match=MatchValue(value=section_names[0]))
                )
            else:
                filter_conditions.append(
                    FieldCondition(key="section_name", match=MatchAny(any=section_names))
                )

        if paper_ids:
            if len(paper_ids) == 1:
                filter_conditions.append(
                    FieldCondition(key="paper_id", match=MatchValue(value=paper_ids[0]))
                )
            else:
                filter_conditions.append(
                    FieldCondition(key="paper_id", match=MatchAny(any=paper_ids))
                )

        query_filter = Filter(must=filter_conditions) if filter_conditions else None

        # Generate sparse vector for query
        sparse_vec = self.bm25_vectorizer.vectorize(query, is_query=True)

        # Use Qdrant's native hybrid search with Prefetch and RRF fusion
        prefetch_limit = limit * 2  # Prefetch more for better fusion

        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    # Dense vector prefetch
                    Prefetch(
                        query=query_embedding,
                        using="",  # Default dense vector
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                    # Sparse vector prefetch
                    Prefetch(
                        query=QdrantSparseVector(
                            indices=sparse_vec.indices,
                            values=sparse_vec.values,
                        ),
                        using="bm25",
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),  # Reciprocal Rank Fusion
                limit=limit,
            )

            return [{'score': r.score, 'hybrid_score': r.score, **r.payload} for r in results.points]

        except Exception as e:
            # Fall back to dense-only if hybrid fails (e.g., no sparse vectors indexed)
            logger.warning(f"Hybrid search failed, falling back to dense: {e}")
            return self.search(
                query_embedding=query_embedding,
                limit=limit,
                chunk_types=chunk_types,
                section_names=section_names,
                paper_ids=paper_ids,
            )

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by its ID using direct point retrieval.

        Uses the deterministic UUID mapping for O(1) lookup instead of scroll.
        """
        point_id = self._chunk_id_to_point_id(chunk_id)

        try:
            # Direct point retrieval is O(1)
            results = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
            )

            if results:
                return results[0].payload
            return None

        except Exception as e:
            logger.warning(f"Failed to retrieve chunk {chunk_id}: {e}")
            # Fall back to scroll with indexed filter
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="_chunk_id", match=MatchValue(value=chunk_id))]
                ),
                limit=1,
            )

            if results and results[0]:
                return results[0][0].payload
            return None

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Batch retrieve multiple chunks by their IDs.

        More efficient than calling get_chunk_by_id multiple times.
        """
        if not chunk_ids:
            return []

        point_ids = [self._chunk_id_to_point_id(cid) for cid in chunk_ids]

        try:
            results = self.client.retrieve(
                collection_name=self.collection_name,
                ids=point_ids,
                with_payload=True,
            )
            return [r.payload for r in results if r.payload]
        except Exception as e:
            logger.warning(f"Batch retrieve failed: {e}")
            # Fall back to individual lookups
            return [p for cid in chunk_ids if (p := self.get_chunk_by_id(cid)) is not None]

    def get_parent_chunk(self, fine_chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get the parent section chunk for a fine chunk.

        Used for context expansion during retrieval.
        """
        parent_id = fine_chunk.get('parent_chunk_id')
        if not parent_id:
            return None

        return self.get_chunk_by_id(parent_id)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        info = self.client.get_collection(self.collection_name)
        return {
            'total_points': info.points_count,
            'status': info.status,
            'vector_dimension': self.embedding_dimension,
        }

    def delete_collection(self) -> bool:
        """Delete the collection (use with caution)."""
        self.client.delete_collection(self.collection_name)
        logger.warning(f"Deleted collection: {self.collection_name}")
        return True

    def get_chunks_by_paper(
        self,
        paper_id: str,
        chunk_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get all chunks for a specific paper."""
        filter_conditions = [
            FieldCondition(key="paper_id", match=MatchValue(value=paper_id))
        ]

        if chunk_types:
            if len(chunk_types) == 1:
                filter_conditions.append(
                    FieldCondition(key="chunk_type", match=MatchValue(value=chunk_types[0]))
                )
            else:
                filter_conditions.append(
                    FieldCondition(key="chunk_type", match=MatchAny(any=chunk_types))
                )

        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(must=filter_conditions),
            limit=1000,  # Assume papers have fewer than 1000 chunks
        )

        return [point.payload for point in results]

    def get_all_papers(self) -> List[Dict[str, Any]]:
        """Get all unique papers with their metadata from the collection.

        Scrolls through the collection and aggregates by paper_id.
        Returns one entry per paper with metadata from the first chunk found.
        """
        papers, _ = self.get_papers_paginated(offset=0, limit=None)
        return papers

    def get_papers_paginated(
        self, offset: int = 0, limit: Optional[int] = 50
    ) -> tuple[List[Dict[str, Any]], int]:
        """Get paginated unique papers with their metadata.

        Scrolls through the collection, deduplicates by paper_id,
        and returns the requested page of papers.

        Args:
            offset: Number of papers to skip
            limit: Maximum number of papers to return (None for all)

        Returns:
            Tuple of (papers list, total paper count)
        """
        papers: Dict[str, Dict[str, Any]] = {}
        scroll_offset = None

        while True:
            results, scroll_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=None,
                limit=1000,
                offset=scroll_offset,
                with_payload=True,
            )

            if not results:
                break

            for point in results:
                payload = point.payload
                paper_id = payload.get('paper_id')
                if not paper_id or paper_id in papers:
                    continue

                papers[paper_id] = {
                    'paper_id': paper_id,
                    'title': payload.get('title', 'Unknown'),
                    'authors': payload.get('authors', []),
                    'year': payload.get('year'),
                    'file_name': payload.get('file_name', ''),
                    'project_tag': payload.get('project_tag'),
                    'research_area': payload.get('research_area'),
                }

            if scroll_offset is None:
                break

        # Convert to list and get total count
        all_papers = list(papers.values())
        total_count = len(all_papers)

        # Apply pagination
        if limit is None:
            return all_papers[offset:], total_count
        return all_papers[offset:offset + limit], total_count

    def get_paper_chunk_stats(self, paper_id: str) -> Dict[str, int]:
        """Get chunk count by type for a specific paper."""
        chunks = self.get_chunks_by_paper(paper_id)
        stats: Dict[str, int] = {}

        for chunk in chunks:
            chunk_type = chunk.get('chunk_type', 'unknown')
            stats[chunk_type] = stats.get(chunk_type, 0) + 1

        return stats

    def delete_paper_chunks(self, paper_id: str) -> int:
        """Delete all chunks for a specific paper.

        Returns the number of points deleted.
        """
        # First count how many we're deleting
        chunks = self.get_chunks_by_paper(paper_id)
        count = len(chunks)

        if count == 0:
            return 0

        # Delete all points with matching paper_id
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
            ),
        )

        logger.info(f"Deleted {count} chunks for paper {paper_id}")
        return count

    def update_paper_chunks_metadata(self, paper_id: str, update_fields: Dict[str, Any]) -> int:
        """Update metadata fields for all chunks of a paper.

        Args:
            paper_id: The paper ID
            update_fields: Dict of field names to new values (e.g., {"title": "New Title", "authors": ["Author 1"]})

        Returns:
            Number of points updated
        """
        from qdrant_client.models import SetPayload

        # Get all chunk IDs for this paper
        chunks = self.get_chunks_by_paper(paper_id)
        if not chunks:
            return 0

        # Convert chunk_ids to Qdrant point IDs (UUIDs)
        point_ids = [self._chunk_id_to_point_id(chunk["chunk_id"]) for chunk in chunks]

        # Update payload for all chunks
        self.client.set_payload(
            collection_name=self.collection_name,
            payload=update_fields,
            points=point_ids,
        )

        logger.info(f"Updated {len(point_ids)} chunks for paper {paper_id} with fields: {list(update_fields.keys())}")
        return len(point_ids)
