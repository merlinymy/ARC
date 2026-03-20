"""Chat service for conversation and message CRUD operations."""

import logging
from typing import Optional
from datetime import datetime

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from database.models import Conversation, Message

logger = logging.getLogger(__name__)


class ChatService:
    """Service for managing conversations and messages."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def list_conversations(self, user_id: int) -> list[Conversation]:
        """List all conversations for a user, ordered by updated_at desc."""
        result = await self.session.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
        )
        return list(result.scalars().all())

    async def get_conversation(
        self,
        conversation_id: str,
        user_id: int,
        include_messages: bool = True
    ) -> Optional[Conversation]:
        """Get a conversation by ID, optionally with messages."""
        query = select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id
        )

        if include_messages:
            query = query.options(selectinload(Conversation.messages))

        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def create_conversation(
        self,
        conversation_id: str,
        user_id: int,
        title: Optional[str] = None
    ) -> Conversation:
        """Create a new conversation."""
        conversation = Conversation(
            id=conversation_id,
            user_id=user_id,
            title=title,
        )
        self.session.add(conversation)
        await self.session.commit()
        await self.session.refresh(conversation)
        logger.info(f"Created conversation: {conversation_id}")
        return conversation

    async def update_conversation(
        self,
        conversation_id: str,
        user_id: int,
        title: Optional[str] = None
    ) -> Optional[Conversation]:
        """Update a conversation's title."""
        result = await self.session.execute(
            update(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id
            )
            .values(title=title, updated_at=datetime.utcnow())
            .returning(Conversation)
        )
        await self.session.commit()
        return result.scalar_one_or_none()

    async def delete_conversation(self, conversation_id: str, user_id: int) -> bool:
        """Delete a conversation and all its messages."""
        result = await self.session.execute(
            delete(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id
            )
        )
        await self.session.commit()
        deleted = result.rowcount > 0
        if deleted:
            logger.info(f"Deleted conversation: {conversation_id}")
        return deleted

    async def add_message(
        self,
        conversation_id: str,
        user_id: int,
        role: str,
        content: str,
        metadata: Optional[dict] = None
    ) -> Message:
        """Add a message to a conversation."""
        # Verify conversation belongs to user
        conversation = await self.get_conversation(
            conversation_id, user_id, include_messages=False
        )
        if conversation is None:
            raise ValueError(f"Conversation not found: {conversation_id}")

        # Log what we're about to persist
        import hashlib
        content_hash = hashlib.md5(content.encode()).hexdigest()
        logger.info(f"[CHAT_SERVICE] Adding message to conversation {conversation_id}")
        logger.info(f"[CHAT_SERVICE] Role: {role}, Length: {len(content)} chars")
        logger.info(f"[CHAT_SERVICE] Content MD5: {content_hash}")
        logger.info(f"[CHAT_SERVICE] Content preview: {content[:150]}...")
        logger.info(f"[CHAT_SERVICE] Has metadata: {metadata is not None}")

        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            message_metadata=metadata,
        )
        self.session.add(message)

        # Update conversation's updated_at
        await self.session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(updated_at=datetime.utcnow())
        )

        await self.session.commit()
        await self.session.refresh(message)

        logger.info(f"[CHAT_SERVICE] Message committed with ID: {message.id}")
        return message

    async def get_messages(
        self,
        conversation_id: str,
        user_id: int,
        limit: Optional[int] = None
    ) -> list[Message]:
        """Get messages for a conversation."""
        # Verify conversation belongs to user
        conversation = await self.get_conversation(
            conversation_id, user_id, include_messages=False
        )
        if conversation is None:
            return []

        query = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
        )

        if limit:
            query = query.limit(limit)

        result = await self.session.execute(query)
        return list(result.scalars().all())
