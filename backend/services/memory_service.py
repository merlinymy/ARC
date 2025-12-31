"""Memory service for user memory CRUD operations."""

import logging
from typing import Optional

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import UserMemory

logger = logging.getLogger(__name__)


class MemoryService:
    """Service for managing user memory facts."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def list_memories(
        self,
        user_id: int,
        category: Optional[str] = None
    ) -> list[UserMemory]:
        """List all memory facts for a user, optionally filtered by category."""
        query = select(UserMemory).where(UserMemory.user_id == user_id)

        if category:
            query = query.where(UserMemory.category == category)

        query = query.order_by(UserMemory.created_at.desc())
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_memory(self, memory_id: int, user_id: int) -> Optional[UserMemory]:
        """Get a specific memory fact by ID."""
        result = await self.session.execute(
            select(UserMemory).where(
                UserMemory.id == memory_id,
                UserMemory.user_id == user_id
            )
        )
        return result.scalar_one_or_none()

    async def add_memory(
        self,
        user_id: int,
        fact: str,
        category: Optional[str] = None
    ) -> UserMemory:
        """Add a new memory fact for the user."""
        memory = UserMemory(
            user_id=user_id,
            fact=fact,
            category=category,
        )
        self.session.add(memory)
        await self.session.commit()
        await self.session.refresh(memory)
        logger.info(f"Added memory for user {user_id}: {fact[:50]}...")
        return memory

    async def delete_memory(self, memory_id: int, user_id: int) -> bool:
        """Delete a memory fact."""
        result = await self.session.execute(
            delete(UserMemory).where(
                UserMemory.id == memory_id,
                UserMemory.user_id == user_id
            )
        )
        await self.session.commit()
        deleted = result.rowcount > 0
        if deleted:
            logger.info(f"Deleted memory {memory_id} for user {user_id}")
        return deleted

    async def get_memory_context(self, user_id: int) -> str:
        """Get formatted memory context for LLM prompts."""
        memories = await self.list_memories(user_id)

        if not memories:
            return ""

        # Group by category
        categorized: dict[str, list[str]] = {}
        for memory in memories:
            cat = memory.category or "general"
            if cat not in categorized:
                categorized[cat] = []
            categorized[cat].append(memory.fact)

        # Format as context string
        lines = ["User context:"]
        for category, facts in categorized.items():
            lines.append(f"  {category.capitalize()}:")
            for fact in facts:
                lines.append(f"    - {fact}")

        return "\n".join(lines)
