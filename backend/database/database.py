"""Database engine and session management."""

import logging
from typing import AsyncGenerator
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select

from .models import Base, User
from config import settings

logger = logging.getLogger(__name__)

# Convert sqlite:/// to sqlite+aiosqlite:/// for async support
database_url = settings.database_url
if database_url.startswith("sqlite:///"):
    database_url = database_url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)

# Ensure the data directory exists
db_path = database_url.replace("sqlite+aiosqlite:///", "")
if db_path.startswith("./"):
    db_dir = Path(db_path).parent
    db_dir.mkdir(parents=True, exist_ok=True)

# Create async engine
engine = create_async_engine(
    database_url,
    echo=settings.environment == "development" and settings.log_level == "DEBUG",
    future=True,
)

# Create async session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database sessions."""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created successfully")


async def create_default_user() -> None:
    """Create the default user if it doesn't exist."""
    from auth.auth import get_password_hash

    async with async_session_maker() as session:
        # Check if user exists
        result = await session.execute(
            select(User).where(User.username == settings.default_username)
        )
        existing_user = result.scalar_one_or_none()

        if existing_user is None:
            # Create default user
            password_hash = get_password_hash(settings.default_password)
            user = User(
                username=settings.default_username,
                password_hash=password_hash,
            )
            session.add(user)
            await session.commit()
            logger.info(f"Created default user: {settings.default_username}")
        else:
            logger.info(f"Default user already exists: {settings.default_username}")
