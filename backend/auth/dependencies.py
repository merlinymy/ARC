"""FastAPI dependencies for authentication."""

from typing import Annotated, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_async_session, User
from .auth import decode_access_token

# HTTP Bearer token scheme
security = HTTPBearer()
security_optional = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> User:
    """Get the current authenticated user from JWT token.

    Raises HTTPException 401 if token is invalid or user not found.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token = credentials.credentials
    payload = decode_access_token(token)

    if payload is None:
        raise credentials_exception

    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception

    # Get user from database
    result = await session.execute(
        select(User).where(User.username == username)
    )
    user = result.scalar_one_or_none()

    if user is None:
        raise credentials_exception

    return user


async def get_current_user_optional(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security_optional)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> Optional[User]:
    """Get the current authenticated user from JWT token, or None if not authenticated.

    Returns None if no token is provided or token is invalid.
    """
    if credentials is None:
        return None

    token = credentials.credentials
    payload = decode_access_token(token)

    if payload is None:
        return None

    username: str = payload.get("sub")
    if username is None:
        return None

    # Get user from database
    result = await session.execute(
        select(User).where(User.username == username)
    )
    return result.scalar_one_or_none()
