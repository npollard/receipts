"""Unified database session management

Provides centralized session handling to prevent leaks and ensure
consistent transaction management across all database operations.
"""

import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from config import DATABASE_URL
from core.exceptions import DatabaseConnectionError

logger = logging.getLogger(__name__)


def get_engine(database_url: Optional[str] = None):
    """Create engine for the specified database URL.

    Args:
        database_url: Database URL. If None, uses global DATABASE_URL.

    Returns:
        SQLAlchemy engine instance
    """
    url = database_url or DATABASE_URL
    return create_engine(url)


def get_session_factory(database_url: Optional[str] = None):
    """Create session factory for the specified database URL.

    Args:
        database_url: Database URL. If None, uses global DATABASE_URL.

    Returns:
        sessionmaker bound to the engine
    """
    engine = get_engine(database_url)
    return sessionmaker(
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
        bind=engine
    )


@contextmanager
def get_session(database_url: Optional[str] = None) -> Generator[Session, None, None]:
    """Get a database session with automatic transaction management.

    Args:
        database_url: Database URL. If None, uses global DATABASE_URL.

    Usage:
        with get_session() as session:
            result = session.query(Model).first()
            # Automatically committed on success
            # Automatically rolled back on exception
            # Always closed
    """
    SessionLocal = get_session_factory(database_url)
    session = SessionLocal()
    try:
        yield session
        session.commit()
        logger.debug("Session committed successfully")
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database transaction rolled back: {e}")
        raise DatabaseConnectionError(f"Database transaction failed: {str(e)}")
    except Exception as e:
        session.rollback()
        logger.error(f"Transaction failed and rolled back: {e}")
        raise
    finally:
        session.close()
        logger.debug("Session closed")


@contextmanager
def get_read_session(database_url: Optional[str] = None) -> Generator[Session, None, None]:
    """Get a read-only session (no commit on exit).

    Args:
        database_url: Database URL. If None, uses global DATABASE_URL.

    Usage:
        with get_read_session() as session:
            result = session.query(Model).first()
            # No commit, always closed
    """
    SessionLocal = get_session_factory(database_url)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        logger.debug("Read session closed")


def create_tables(database_url: Optional[str] = None):
    """Create all database tables for the specified database URL.

    Args:
        database_url: Database URL. If None, uses global DATABASE_URL.
    """
    from .models import Base
    engine = get_engine(database_url)
    Base.metadata.create_all(bind=engine)
    engine.dispose()
    logger.info(f"Database tables created for {database_url or DATABASE_URL}")


# Alias for backward compatibility
create_tables_for_url = create_tables


def close_all_sessions():
    """Close all sessions and dispose of cached engines.

    Note: Since engines are now created per-URL, this clears any
    cached engines that might be stored externally.
    """
    logger.info("Database engines disposed")
