"""Database connection management and transaction helpers"""

import logging
from contextlib import contextmanager
from typing import Optional, Generator, Any, Callable
from functools import wraps
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from config import DATABASE_URL

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Centralized database connection management"""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or DATABASE_URL
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()

    def create_tables(self):
        """Create all database tables"""
        from database_models import Base
        Base.metadata.create_all(bind=self.engine)

    def close(self):
        """Close database connections"""
        self.engine.dispose()

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Context manager for automatic session handling"""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database transaction rolled back: {e}")
            raise
        finally:
            session.close()

    @contextmanager
    def transaction_scope(self) -> Generator[Session, None, None]:
        """Context manager for explicit transaction control"""
        session = self.get_session()
        try:
            yield session
        except Exception as e:
            session.rollback()
            logger.error(f"Transaction failed and rolled back: {e}")
            raise
        finally:
            session.close()


def with_database_session(func: Callable) -> Callable:
    """Decorator to automatically provide database session"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, 'db_connection'):
            with self.db_connection.session_scope() as session:
                return func(self, session, *args, **kwargs)
        else:
            raise AttributeError(f"{self.__class__.__name__} missing db_connection attribute")
    return wrapper


def with_transaction(func: Callable) -> Callable:
    """Decorator to automatically handle transactions"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, 'db_connection'):
            with self.db_connection.transaction_scope() as session:
                return func(self, session, *args, **kwargs)
        else:
            raise AttributeError(f"{self.__class__.__name__} missing db_connection attribute")
    return wrapper


class DatabaseQueryHelper:
    """Helper class for common database query patterns"""

    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, model_class, entity_id: Any, user_id: Optional[Any] = None):
        """Generic get by ID with optional user filtering"""
        query = self.session.query(model_class).filter(model_class.id == entity_id)
        if user_id and hasattr(model_class, 'user_id'):
            query = query.filter(model_class.user_id == user_id)
        return query.first()

    def get_by_field(self, model_class, field_name: str, field_value: Any,
                    user_id: Optional[Any] = None):
        """Generic get by field with optional user filtering"""
        field = getattr(model_class, field_name)
        query = self.session.query(model_class).filter(field == field_value)
        if user_id and hasattr(model_class, 'user_id'):
            query = query.filter(model_class.user_id == user_id)
        return query.first()

    def get_paginated(self, model_class, limit: int = 50, offset: int = 0,
                     user_id: Optional[Any] = None, status_filter: Optional[str] = None,
                     order_by_field: str = 'created_at', descending: bool = True):
        """Generic paginated query with optional filtering"""
        query = self.session.query(model_class)

        # Apply user filter if model has user_id
        if user_id and hasattr(model_class, 'user_id'):
            query = query.filter(model_class.user_id == user_id)

        # Apply status filter if provided and model has status
        if status_filter and hasattr(model_class, 'status'):
            query = query.filter(model_class.status == status_filter)

        # Apply ordering
        order_field = getattr(model_class, order_by_field)
        if descending:
            query = query.order_by(order_field.desc())
        else:
            query = query.order_by(order_field.asc())

        # Apply pagination
        return query.offset(offset).limit(limit).all()

    def count_records(self, model_class, user_id: Optional[Any] = None,
                     status_filter: Optional[str] = None):
        """Generic count with optional filtering"""
        query = self.session.query(model_class)

        if user_id and hasattr(model_class, 'user_id'):
            query = query.filter(model_class.user_id == user_id)

        if status_filter and hasattr(model_class, 'status'):
            query = query.filter(model_class.status == status_filter)

        return query.count()

    def delete_by_id(self, model_class, entity_id: Any, user_id: Optional[Any] = None) -> bool:
        """Generic delete by ID with optional user filtering"""
        entity = self.get_by_id(model_class, entity_id, user_id)
        if entity:
            self.session.delete(entity)
            return True
        return False


class DatabaseTransactionHelper:
    """Helper class for transaction operations"""

    def __init__(self, session: Session):
        self.session = session

    def create_and_commit(self, model_class, **kwargs) -> Any:
        """Create entity and commit in one operation"""
        entity = model_class(**kwargs)
        self.session.add(entity)
        self.session.flush()  # Get ID without committing
        return entity

    def update_and_commit(self, entity, **kwargs) -> Any:
        """Update entity and commit in one operation"""
        for key, value in kwargs.items():
            if hasattr(entity, key):
                setattr(entity, key, value)
        self.session.flush()
        return entity

    def bulk_create(self, model_class, entities_data: list[dict]) -> list[Any]:
        """Bulk create entities"""
        entities = [model_class(**data) for data in entities_data]
        self.session.bulk_save_objects(entities)
        self.session.flush()
        return entities

    def execute_raw_query(self, query: str, params: Optional[dict] = None) -> Any:
        """Execute raw SQL query"""
        return self.session.execute(query, params or {})


def handle_uuid_for_db(uuid_value: Any) -> Any:
    """Convert UUID to appropriate format for database"""
    from uuid import UUID
    if isinstance(uuid_value, UUID):
        # Check if using SQLite
        if DATABASE_URL.startswith("sqlite"):
            return str(uuid_value)  # SQLite uses string
        return uuid_value  # PostgreSQL uses UUID
    return uuid_value


def get_database_type() -> str:
    """Get the current database type"""
    if DATABASE_URL.startswith("sqlite"):
        return "sqlite"
    elif DATABASE_URL.startswith("postgresql"):
        return "postgresql"
    else:
        return "unknown"


# Global connection registry for singleton pattern
_connection_registry = {}


def get_database_connection(database_url: Optional[str] = None) -> DatabaseConnection:
    """Get or create database connection (singleton pattern)"""
    if database_url is None:
        database_url = DATABASE_URL

    if database_url not in _connection_registry:
        _connection_registry[database_url] = DatabaseConnection(database_url)
        logger.info(f"Created database connection: {database_url}")

    return _connection_registry[database_url]


def close_all_connections():
    """Close all database connections"""
    for connection in _connection_registry.values():
        connection.close()
    _connection_registry.clear()
    logger.info("Closed all database connections")
