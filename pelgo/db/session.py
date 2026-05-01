from __future__ import annotations

import os

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

_engine: AsyncEngine | None = None
_session_maker: async_sessionmaker | None = None


def _get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        url = os.environ.get(
            "DATABASE_URL",
            "postgresql+asyncpg://pelgo:pelgo@localhost:5432/pelgo",
        )
        _engine = create_async_engine(url, pool_size=5, max_overflow=5, echo=False)
    return _engine


def _get_session_maker() -> async_sessionmaker:
    global _session_maker
    if _session_maker is None:
        _session_maker = async_sessionmaker(
            _get_engine(), expire_on_commit=False, class_=AsyncSession
        )
    return _session_maker


def AsyncSessionLocal() -> AsyncSession:
    return _get_session_maker()()


def reset_for_testing(url: str) -> None:
    """Replace the engine/session-maker singletons (test use only)."""
    global _engine, _session_maker
    _engine = create_async_engine(url, pool_size=2, max_overflow=2, echo=False)
    _session_maker = async_sessionmaker(
        _engine, expire_on_commit=False, class_=AsyncSession
    )
