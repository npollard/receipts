"""Infrastructure database module for persistence operations"""

from .models import (
    Base,
    User,
    Receipt,
    ReceiptItem,
    parse_receipt_date,
    extract_merchant_name,
)

from .repository import (
    DatabaseConnection,
    DatabaseConnection as DatabaseManager,  # Backward compatibility
    DatabaseConnection as DatabaseTransactionHelper,  # Backward compatibility
    DatabaseConnection as DatabaseQueryHelper,  # Backward compatibility
    ReceiptRepository,
    handle_uuid_for_db,
    with_database_session,
    with_transaction,
    get_database_connection,
    close_all_connections,
)

from .user_repository import (
    UserRepository,
)

from .session import (
    get_session,
    get_read_session,
    create_tables,
    create_tables_for_url,
    close_all_sessions,
)

from .mappers import (
    domain_to_orm,
    orm_to_domain,
    update_orm_from_domain,
    domain_item_to_orm,
    orm_item_to_domain,
    receipt_to_dict,
)

__all__ = [
    # Models
    'Base',
    'User',
    'Receipt',
    'ReceiptItem',
    'DatabaseManager',
    'parse_receipt_date',
    'extract_merchant_name',
    # Repository
    'DatabaseConnection',
    'DatabaseTransactionHelper',
    'DatabaseQueryHelper',
    'ReceiptRepository',
    'UserRepository',
    'handle_uuid_for_db',
    'with_database_session',
    'with_transaction',
    'get_database_connection',
    'close_all_connections',
    # Session
    'get_session',
    'get_read_session',
    'create_tables',
    'create_tables_for_url',
    'close_all_sessions',
    # Mappers
    'domain_to_orm',
    'orm_to_domain',
    'update_orm_from_domain',
    'domain_item_to_orm',
    'orm_item_to_domain',
    'receipt_to_dict',
]
