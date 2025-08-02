# utils/__init__.py
"""
Utilities Package

Contains utility functions, exception classes, logging configuration,
and helper functions used throughout the application.
"""

from .logger import (
    get_logger, 
    setup_logging,
    setup_application_logging,
    log_execution_time,
    log_function_call,
    PerformanceLogger
)
from .exceptions import (
    ContentModerationError,
    PDFProcessingError,
    ImageExtractionError,
    TextExtractionError,
    ModelLoadingError,
    VisionAnalysisError,
    NLPAnalysisError,
    ValidationError,
    FileUploadError,
    FileSizeError,
    FileTypeError,
    handle_exception,
    create_error_response
)
from .helpers import (
    allowed_file,
    secure_filename,
    generate_file_hash,
    get_file_info,
    create_unique_filename,
    format_file_size,
    clean_text,
    extract_text_keywords,
    validate_json_data,
    truncate_text,
    safe_json_loads,
    safe_json_dumps
)

__all__ = [
    # Logger exports
    'get_logger',
    'setup_logging', 
    'setup_application_logging',
    'log_execution_time',
    'log_function_call',
    'PerformanceLogger',
    
    # Exception exports
    'ContentModerationError',
    'PDFProcessingError',
    'ImageExtractionError', 
    'TextExtractionError',
    'ModelLoadingError',
    'VisionAnalysisError',
    'NLPAnalysisError',
    'ValidationError',
    'FileUploadError',
    'FileSizeError',
    'FileTypeError',
    'handle_exception',
    'create_error_response',
    
    # Helper exports
    'allowed_file',
    'secure_filename',
    'generate_file_hash',
    'get_file_info',
    'create_unique_filename',
    'format_file_size',
    'clean_text',
    'extract_text_keywords',
    'validate_json_data',
    'truncate_text',
    'safe_json_loads',
    'safe_json_dumps'
]