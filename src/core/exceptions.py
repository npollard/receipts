"""Custom exceptions for receipt processing pipeline"""

class ReceiptProcessingError(Exception):
    """Base exception for receipt processing errors"""
    pass

class OCRError(ReceiptProcessingError):
    """Exception for OCR-related errors"""
    pass

class ImageProcessingError(OCRError):
    """Exception for image processing errors"""
    pass

class TextExtractionError(OCRError):
    """Exception for text extraction errors"""
    pass

class VisionAPIError(OCRError):
    """Exception for Vision API errors"""
    pass

class QualityScoreError(OCRError):
    """Exception for OCR quality scoring errors"""
    pass

class ParsingError(ReceiptProcessingError):
    """Exception for receipt parsing errors"""
    pass

class ValidationError(ParsingError):
    """Exception for data validation errors"""
    pass

class AIModelError(ParsingError):
    """Exception for AI model errors"""
    pass

class TokenUsageError(ParsingError):
    """Exception for token usage tracking errors"""
    pass

class StorageError(ReceiptProcessingError):
    """Exception for storage/database errors"""
    pass

class DatabaseConnectionError(StorageError):
    """Exception for database connection errors"""
    pass

class DataIntegrityError(StorageError):
    """Exception for data integrity errors"""
    pass

class IdempotencyError(StorageError):
    """Exception for idempotency handling errors"""
    pass

class BatchProcessingError(ReceiptProcessingError):
    """Exception for batch processing errors"""
    pass

class FileOperationError(BatchProcessingError):
    """Exception for file operation errors"""
    pass

class RetryExhaustedError(ReceiptProcessingError):
    """Exception when retry attempts are exhausted"""
    pass

class ConfigurationError(ReceiptProcessingError):
    """Exception for configuration errors"""
    pass

class AuthenticationError(ReceiptProcessingError):
    """Exception for authentication errors"""
    pass

class RateLimitError(ReceiptProcessingError):
    """Exception for rate limiting errors"""
    pass
