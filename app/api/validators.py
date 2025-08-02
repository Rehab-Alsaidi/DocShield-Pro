# app/api/validators.py
import os
import mimetypes
from typing import Dict, Any
from werkzeug.datastructures import FileStorage

from app.config import app_config
from utils.helpers import allowed_file

def validate_upload_request(request) -> Dict[str, Any]:
    """Validate file upload request"""
    
    # Check if file is in request
    if 'file' not in request.files:
        return {
            'valid': False,
            'message': 'No file provided in request'
        }
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        return {
            'valid': False,
            'message': 'No file selected'
        }
    
    return {'valid': True, 'message': 'Request valid'}

def validate_file(file: FileStorage) -> Dict[str, Any]:
    """Validate uploaded file"""
    
    # Check file size (Flask handles this automatically, but we can add custom logic)
    if hasattr(file, 'content_length') and file.content_length:
        if file.content_length > app_config.max_file_size:
            return {
                'valid': False,
                'message': f'File too large. Maximum size is {app_config.max_file_size // (1024*1024)}MB'
            }
    
    # Check file extension
    if not allowed_file(file.filename, app_config.allowed_extensions):
        return {
            'valid': False,
            'message': f'Invalid file type. Allowed extensions: {", ".join(app_config.allowed_extensions)}'
        }
    
    # Check MIME type
    if file.mimetype and not file.mimetype.startswith('application/pdf'):
        return {
            'valid': False,
            'message': 'Invalid file type. Only PDF files are allowed.'
        }
    
    # Additional file content validation
    try:
        # Read first few bytes to verify PDF signature
        file.seek(0)
        header = file.read(8)
        file.seek(0)  # Reset file pointer
        
        if not header.startswith(b'%PDF-'):
            return {
                'valid': False,
                'message': 'File does not appear to be a valid PDF'
            }
    except Exception:
        return {
            'valid': False,
            'message': 'Unable to validate file content'
        }
    
    return {'valid': True, 'message': 'File valid'}

def validate_text_analysis_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate text analysis request"""
    
    if not data:
        return {
            'valid': False,
            'message': 'No data provided'
        }
    
    if 'text' not in data:
        return {
            'valid': False,
            'message': 'Text field required'
        }
    
    text = data['text']
    
    if not isinstance(text, str):
        return {
            'valid': False,
            'message': 'Text must be a string'
        }
    
    if len(text.strip()) < 10:
        return {
            'valid': False,
            'message': 'Text too short. Minimum 10 characters required.'
        }
    
    if len(text) > 100000:  # 100KB text limit
        return {
            'valid': False,
            'message': 'Text too long. Maximum 100,000 characters allowed.'
        }
    
    return {'valid': True, 'message': 'Text valid'}

def validate_confidence_threshold(threshold_str: str) -> Dict[str, Any]:
    """Validate confidence threshold parameter"""
    
    try:
        threshold = float(threshold_str)
        
        if threshold < 0.0 or threshold > 1.0:
            return {
                'valid': False,
                'message': 'Confidence threshold must be between 0.0 and 1.0'
            }
        
        return {
            'valid': True,
            'value': threshold,
            'message': 'Threshold valid'
        }
        
    except ValueError:
        return {
            'valid': False,
            'message': 'Confidence threshold must be a valid number'
        }

def validate_processing_options(form_data) -> Dict[str, Any]:
    """Validate processing options from form data"""
    
    options = {}
    
    # Include images option
    include_images = form_data.get('include_images', 'true').lower()
    if include_images not in ['true', 'false']:
        return {
            'valid': False,
            'message': 'include_images must be true or false'
        }
    options['include_images'] = include_images == 'true'
    
    # Include text option
    include_text = form_data.get('include_text', 'true').lower()
    if include_text not in ['true', 'false']:
        return {
            'valid': False,
            'message': 'include_text must be true or false'
        }
    options['include_text'] = include_text == 'true'
    
    # Confidence threshold
    threshold_str = form_data.get('confidence_threshold', '0.7')
    threshold_validation = validate_confidence_threshold(threshold_str)
    if not threshold_validation['valid']:
        return threshold_validation
    
    options['confidence_threshold'] = threshold_validation['value']
    
    return {
        'valid': True,
        'options': options,
        'message': 'Options valid'
    }

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    
    # Remove path components
    filename = os.path.basename(filename)
    
    # Replace unsafe characters
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    sanitized = ''.join(c if c in safe_chars else '_' for c in filename)
    
    # Ensure it's not empty and has reasonable length
    if not sanitized or sanitized.replace('_', '').replace('.', '') == '':
        sanitized = 'uploaded_file.pdf'
    
    # Limit length
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:250] + ext
    
    return sanitized

def validate_result_id(result_id: str) -> Dict[str, Any]:
    """Validate result ID format"""
    
    if not result_id:
        return {
            'valid': False,
            'message': 'Result ID required'
        }
    
    # Check if it looks like a UUID
    import re
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    
    if not uuid_pattern.match(result_id):
        return {
            'valid': False,
            'message': 'Invalid result ID format'
        }
    
    return {'valid': True, 'message': 'Result ID valid'}

def validate_pagination_params(args) -> Dict[str, Any]:
    """Validate pagination parameters"""
    
    try:
        page = int(args.get('page', 1))
        per_page = int(args.get('per_page', 20))
        
        if page < 1:
            return {
                'valid': False,
                'message': 'Page number must be >= 1'
            }
        
        if per_page < 1 or per_page > 100:
            return {
                'valid': False,
                'message': 'Per page must be between 1 and 100'
            }
        
        return {
            'valid': True,
            'page': page,
            'per_page': per_page,
            'message': 'Pagination valid'
        }
        
    except ValueError:
        return {
            'valid': False,
            'message': 'Invalid pagination parameters'
        }