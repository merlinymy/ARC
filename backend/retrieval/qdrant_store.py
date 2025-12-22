"""Qdrant vector store operations."""

import logging
import hashlib
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)

logger = logging.getLogger(__name__)


class QdrantStore:
    """Qdrant vector store for research paper chunks.

    Single collection with all 6 chunk types, differentiated by metadata.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "research_papers",
        embedding_dimension: int = 1024,
    ):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension

        logger.info(f"Connected to Qdrant at {host}:{port}")

    def ensure_collection(self) -> bool:
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {self.collection_name}")
            return True

        logger.info(f"Collection {self.collection_name} already exists")
        return False

    def upsert_chunks(
        self,
        chunk_ids: List[str],
        embeddings: List[List[float]],
        payloads: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """Upsert chunks to collection."""
        total = 0

        for i in range(0, len(chunk_ids), batch_size):
            batch_ids = chunk_ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_payloads = payloads[i:i + batch_size]

            points = [
                PointStruct(
                    id=int(hashlib.md5(chunk_id.encode()).hexdigest(), 16) % (2**63),
                    vector=embedding,
                    payload={**payload, '_chunk_id': chunk_id}
                )
                for chunk_id, embedding, payload
                in zip(batch_ids, batch_embeddings, batch_payloads)
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            total += len(points)

        logger.info(f"Upserted {total} chunks to {self.collection_name}")
        return total

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
    ) -> List[Dict[str, Any]]:
        """Search using a retrieval strategy."""
        all_results = []

        for chunk_type in chunk_types:
            results = self.search(
                query_embedding=query_embedding,
                limit=top_k,
                chunk_types=[chunk_type],
                section_names=section_filter,
            )
            all_results.extend(results)

        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by its ID."""
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="_chunk_id", match=MatchValue(value=chunk_id))]
            ),
            limit=1,
        )

        if results and results[0]:
            point = results[0][0]
            return point.payload
        return None

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
