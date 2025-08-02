# utils/exceptions.py
"""
Custom exceptions for the PDF Content Moderation System
"""

class ContentModerationError(Exception):
    """Base exception for content moderation errors"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        self.message = message
        self.error_code = error_code or "MODERATION_ERROR"
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> dict:
        """Convert exception to dictionary for JSON serialization"""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'details': self.details
        }

class PDFProcessingError(ContentModerationError):
    """Exception for PDF processing errors"""
    
    def __init__(self, message: str, file_path: str = None, page_number: int = None):
        super().__init__(
            message=message,
            error_code="PDF_PROCESSING_ERROR",
            details={
                'file_path': file_path,
                'page_number': page_number
            }
        )

class ImageExtractionError(PDFProcessingError):
    """Exception for image extraction errors"""
    
    def __init__(self, message: str, file_path: str = None, page_number: int = None, image_index: int = None):
        super().__init__(
            message=message,
            file_path=file_path,
            page_number=page_number
        )
        self.error_code = "IMAGE_EXTRACTION_ERROR"
        self.details['image_index'] = image_index

class TextExtractionError(PDFProcessingError):
    """Exception for text extraction errors"""
    
    def __init__(self, message: str, file_path: str = None, page_number: int = None):
        super().__init__(
            message=message,
            file_path=file_path,
            page_number=page_number
        )
        self.error_code = "TEXT_EXTRACTION_ERROR"

class ModelLoadingError(ContentModerationError):
    """Exception for AI model loading errors"""
    
    def __init__(self, message: str, model_name: str = None, model_type: str = None):
        super().__init__(
            message=message,
            error_code="MODEL_LOADING_ERROR",
            details={
                'model_name': model_name,
                'model_type': model_type
            }
        )

class VisionAnalysisError(ContentModerationError):
    """Exception for computer vision analysis errors"""
    
    def __init__(self, message: str, image_info: dict = None, model_name: str = None):
        super().__init__(
            message=message,
            error_code="VISION_ANALYSIS_ERROR",
            details={
                'image_info': image_info,
                'model_name': model_name
            }
        )

class NLPAnalysisError(ContentModerationError):
    """Exception for NLP analysis errors"""
    
    def __init__(self, message: str, text_length: int = None, language: str = None):
        super().__init__(
            message=message,
            error_code="NLP_ANALYSIS_ERROR",
            details={
                'text_length': text_length,
                'language': language
            }
        )

class ValidationError(ContentModerationError):
    """Exception for input validation errors"""
    
    def __init__(self, message: str, field_name: str = None, validation_type: str = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details={
                'field_name': field_name,
                'validation_type': validation_type
            }
        )

class FileUploadError(ContentModerationError):
    """Exception for file upload errors"""
    
    def __init__(self, message: str, filename: str = None, file_size: int = None, mime_type: str = None):
        super().__init__(
            message=message,
            error_code="FILE_UPLOAD_ERROR",
            details={
                'filename': filename,
                'file_size': file_size,
                'mime_type': mime_type
            }
        )

class FileSizeError(FileUploadError):
    """Exception for file size validation errors"""
    
    def __init__(self, message: str, filename: str = None, actual_size: int = None, max_size: int = None):
        super().__init__(
            message=message,
            filename=filename,
            file_size=actual_size
        )
        self.error_code = "FILE_SIZE_ERROR"
        self.details.update({
            'actual_size': actual_size,
            'max_size': max_size,
            'size_exceeded_by': actual_size - max_size if actual_size and max_size else None
        })

class FileTypeError(FileUploadError):
    """Exception for file type validation errors"""
    
    def __init__(self, message: str, filename: str = None, actual_type: str = None, allowed_types: list = None):
        super().__init__(
            message=message,
            filename=filename,
            mime_type=actual_type
        )
        self.error_code = "FILE_TYPE_ERROR"
        self.details.update({
            'actual_type': actual_type,
            'allowed_types': allowed_types
        })

class ConfigurationError(ContentModerationError):
    """Exception for configuration errors"""
    
    def __init__(self, message: str, config_key: str = None, config_value: str = None):
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details={
                'config_key': config_key,
                'config_value': config_value
            }
        )

class DatabaseError(ContentModerationError):
    """Exception for database operation errors"""
    
    def __init__(self, message: str, operation: str = None, table_name: str = None):
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            details={
                'operation': operation,
                'table_name': table_name
            }
        )

class APIError(ContentModerationError):
    """Exception for API-related errors"""
    
    def __init__(self, message: str, status_code: int = None, endpoint: str = None):
        super().__init__(
            message=message,
            error_code="API_ERROR",
            details={
                'status_code': status_code,
                'endpoint': endpoint
            }
        )

class ResourceNotFoundError(ContentModerationError):
    """Exception for resource not found errors"""
    
    def __init__(self, message: str, resource_type: str = None, resource_id: str = None):
        super().__init__(
            message=message,
            error_code="RESOURCE_NOT_FOUND",
            details={
                'resource_type': resource_type,
                'resource_id': resource_id
            }
        )

class ProcessingTimeoutError(ContentModerationError):
    """Exception for processing timeout errors"""
    
    def __init__(self, message: str, timeout_seconds: int = None, operation: str = None):
        super().__init__(
            message=message,
            error_code="PROCESSING_TIMEOUT",
            details={
                'timeout_seconds': timeout_seconds,
                'operation': operation
            }
        )

class InsufficientResourcesError(ContentModerationError):
    """Exception for insufficient system resources"""
    
    def __init__(self, message: str, resource_type: str = None, required: str = None, available: str = None):
        super().__init__(
            message=message,
            error_code="INSUFFICIENT_RESOURCES",
            details={
                'resource_type': resource_type,
                'required': required,
                'available': available
            }
        )

class ConcurrencyError(ContentModerationError):
    """Exception for concurrency-related errors"""
    
    def __init__(self, message: str, resource_id: str = None, conflicting_operation: str = None):
        super().__init__(
            message=message,
            error_code="CONCURRENCY_ERROR",
            details={
                'resource_id': resource_id,
                'conflicting_operation': conflicting_operation
            }
        )

# Utility functions for exception handling

def handle_exception(func):
    """Decorator to handle and log exceptions"""
    from functools import wraps
    from utils.logger import get_logger
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        try:
            return func(*args, **kwargs)
        except ContentModerationError as e:
            logger.error(f"Content moderation error in {func.__name__}: {e.message}")
            logger.debug(f"Error details: {e.details}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise ContentModerationError(
                message=f"Unexpected error in {func.__name__}: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                details={'function_name': func.__name__, 'original_error': str(e)}
            )
    
    return wrapper

def create_error_response(exception: ContentModerationError, include_details: bool = True) -> dict:
    """Create standardized error response from exception"""
    response = {
        'success': False,
        'error': {
            'type': exception.__class__.__name__,
            'message': exception.message,
            'code': exception.error_code
        }
    }
    
    if include_details and exception.details:
        response['error']['details'] = exception.details
    
    return response

def log_exception_with_context(logger, exception: Exception, context: dict = None):
    """Log exception with additional context"""
    import traceback
    
    if isinstance(exception, ContentModerationError):
        logger.error(f"Content Moderation Error: {exception.message}")
        logger.error(f"Error Code: {exception.error_code}")
        if exception.details:
            logger.error(f"Error Details: {exception.details}")
    else:
        logger.error(f"Unexpected Error: {str(exception)}")
    
    if context:
        logger.error(f"Context: {context}")
    
    # Log full traceback for debugging
    logger.debug(f"Full traceback: {traceback.format_exc()}")

def validate_and_raise(condition: bool, error_class: type, message: str, **kwargs):
    """Validate condition and raise specific exception if false"""
    if not condition:
        raise error_class(message, **kwargs)

# Exception mapping for HTTP status codes
EXCEPTION_STATUS_MAPPING = {
    ValidationError: 400,
    FileUploadError: 400,
    FileSizeError: 413,
    FileTypeError: 400,
    ResourceNotFoundError: 404,
    ProcessingTimeoutError: 408,
    InsufficientResourcesError: 507,
    ConcurrencyError: 409,
    APIError: 500,
    DatabaseError: 500,
    ConfigurationError: 500,
    ContentModerationError: 500
}

def get_http_status_for_exception(exception: Exception) -> int:
    """Get appropriate HTTP status code for exception"""
    for exc_type, status_code in EXCEPTION_STATUS_MAPPING.items():
        if isinstance(exception, exc_type):
            return status_code
    return 500  # Default to internal server error