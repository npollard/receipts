"""File utilities for common file operations"""

import os
import base64
from pathlib import Path
from typing import Union, Optional, BinaryIO, Iterator


def read_file_bytes(file_path: Union[str, Path], chunk_size: int = 4096) -> Iterator[bytes]:
    """Read file in chunks as bytes iterator
    
    Args:
        file_path: Path to the file
        chunk_size: Size of each chunk in bytes
        
    Yields:
        Bytes chunks from the file
    """
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            yield chunk


def read_file_text(file_path: Union[str, Path], encoding: str = 'utf-8') -> str:
    """Read entire file as text
    
    Args:
        file_path: Path to the file
        encoding: File encoding (default: utf-8)
        
    Returns:
        File content as string
    """
    with open(file_path, "r", encoding=encoding) as f:
        return f.read()


def read_file_binary(file_path: Union[str, Path]) -> bytes:
    """Read entire file as binary data
    
    Args:
        file_path: Path to the file
        
    Returns:
        File content as bytes
    """
    with open(file_path, "rb") as f:
        return f.read()


def write_file_text(file_path: Union[str, Path], content: str, encoding: str = 'utf-8') -> None:
    """Write text content to file
    
    Args:
        file_path: Path to the file
        content: Text content to write
        encoding: File encoding (default: utf-8)
    """
    # Create directory if it doesn't exist
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, "w", encoding=encoding) as f:
        f.write(content)


def write_file_binary(file_path: Union[str, Path], content: bytes) -> None:
    """Write binary content to file
    
    Args:
        file_path: Path to the file
        content: Binary content to write
    """
    # Create directory if it doesn't exist
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, "wb") as f:
        f.write(content)


def file_exists(file_path: Union[str, Path]) -> bool:
    """Check if file exists
    
    Args:
        file_path: Path to check
        
    Returns:
        True if file exists, False otherwise
    """
    return Path(file_path).exists()


def get_file_size(file_path: Union[str, Path]) -> int:
    """Get file size in bytes
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in bytes
    """
    return Path(file_path).stat().st_size


def ensure_directory(dir_path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary
    
    Args:
        dir_path: Path to the directory
        
    Returns:
        Path object of the directory
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_extension(file_path: Union[str, Path]) -> str:
    """Get file extension including the dot
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension (e.g., '.jpg', '.png')
    """
    return Path(file_path).suffix.lower()


def encode_file_base64(file_path: Union[str, Path]) -> str:
    """Encode file content as base64 string
    
    Args:
        file_path: Path to the file
        
    Returns:
        Base64 encoded string
    """
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def decode_base64_to_file(base64_data: str, file_path: Union[str, Path]) -> None:
    """Decode base64 data and save to file
    
    Args:
        base64_data: Base64 encoded string
        file_path: Path where to save the decoded file
    """
    # Create directory if it doesn't exist
    ensure_directory(Path(file_path).parent)
    
    with open(file_path, "wb") as f:
        f.write(base64.b64decode(base64_data))


def find_files_by_extension(directory: Union[str, Path], 
                           extensions: Union[str, list[str]], 
                           recursive: bool = True) -> list[Path]:
    """Find files by extension in directory
    
    Args:
        directory: Directory to search
        extensions: Single extension or list of extensions (with or without dot)
        recursive: Whether to search recursively
        
    Returns:
        List of file paths
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    
    # Normalize extensions to include dot
    if isinstance(extensions, str):
        extensions = [extensions]
    
    normalized_exts = []
    for ext in extensions:
        if not ext.startswith('.'):
            ext = '.' + ext
        normalized_exts.append(ext.lower())
    
    # Find files
    pattern = "**/*" if recursive else "*"
    files = []
    for file_path in dir_path.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in normalized_exts:
            files.append(file_path)
    
    return sorted(files)


def get_image_files(directory: Union[str, Path], recursive: bool = True) -> list[Path]:
    """Get all image files from directory
    
    Args:
        directory: Directory to search
        recursive: Whether to search recursively
        
    Returns:
        List of image file paths
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif']
    return find_files_by_extension(directory, image_extensions, recursive)
