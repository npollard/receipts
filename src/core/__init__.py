"""Core utilities module for fundamental operations"""

from .file_operations import (
    read_file_bytes,
    read_file_text,
    read_file_binary,
    write_file_text,
    write_file_binary,
    file_exists,
    get_file_size,
    ensure_directory,
    get_file_extension,
    encode_file_base64,
    decode_base64_to_file,
    find_files_by_extension,
    get_image_files
)

from .hashing import (
    calculate_file_hash,
    calculate_image_hash, 
    calculate_data_hash,
    calculate_receipt_data_hash,
    normalize_data_for_hashing,
    calculate_string_hash
)

__all__ = [
    # File operations
    'read_file_bytes',
    'read_file_text',
    'read_file_binary',
    'write_file_text',
    'write_file_binary',
    'file_exists',
    'get_file_size',
    'ensure_directory',
    'get_file_extension',
    'encode_file_base64',
    'decode_base64_to_file',
    'find_files_by_extension',
    'get_image_files',
    
    # Hashing utilities
    'calculate_file_hash',
    'calculate_image_hash',
    'calculate_data_hash', 
    'calculate_receipt_data_hash',
    'normalize_data_for_hashing',
    'calculate_string_hash'
]
