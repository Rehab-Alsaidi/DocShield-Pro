# utils/helpers.py
import os
import hashlib
import mimetypes
import uuid
import base64
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import re

def allowed_file(filename: str, allowed_extensions: list) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def secure_filename(filename: str) -> str:
    """Create secure filename"""
    # Remove path components
    filename = os.path.basename(filename)
    
    # Replace unsafe characters
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    filename = ''.join(c if c in safe_chars else '_' for c in filename)
    
    # Ensure it's not empty
    if not filename or filename.replace('_', '').replace('.', '') == '':
        filename = 'file.pdf'
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:250] + ext
    
    return filename

def generate_file_hash(file_path: str) -> str:
    """Generate SHA256 hash of file"""
    sha256_hash = hashlib.sha256()
    
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except Exception:
        return None

def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get comprehensive file information"""
    if not os.path.exists(file_path):
        return {}
    
    stat = os.stat(file_path)
    
    return {
        'filename': os.path.basename(file_path),
        'size_bytes': stat.st_size,
        'size_mb': round(stat.st_size / (1024 * 1024), 2),
        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'mime_type': mimetypes.guess_type(file_path)[0],
        'file_hash': generate_file_hash(file_path)
    }

def create_unique_filename(original_filename: str, directory: str = None) -> str:
    """Create unique filename to avoid conflicts"""
    secure_name = secure_filename(original_filename)
    name, ext = os.path.splitext(secure_name)
    
    unique_id = str(uuid.uuid4())[:8]
    unique_filename = f"{name}_{unique_id}{ext}"
    
    # Check if directory provided and file exists
    if directory and os.path.exists(os.path.join(directory, unique_filename)):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{name}_{unique_id}_{timestamp}{ext}"
    
    return unique_filename

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def extract_text_keywords(text: str, min_length: int = 3, max_keywords: int = 20) -> List[str]:
    """Extract keywords from text"""
    if not text:
        return []
    
    # Simple keyword extraction
    words = re.findall(r'\b[a-zA-Z]{' + str(min_length) + ',}\b', text.lower())
    
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'this', 'that', 'these', 'those', 'they', 'them', 'their', 'there',
        'where', 'when', 'who', 'what', 'why', 'how', 'can', 'may', 'might'
    }
    
    keywords = [word for word in words if word not in stop_words]
    
    # Return most frequent keywords
    from collections import Counter
    word_counts = Counter(keywords)
    return [word for word, count in word_counts.most_common(max_keywords)]

def validate_json_data(data: Any, required_fields: List[str]) -> Dict[str, Any]:
    """Validate JSON data structure"""
    if not isinstance(data, dict):
        return {
            'valid': False,
            'message': 'Data must be a JSON object'
        }
    
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return {
            'valid': False,
            'message': f'Missing required fields: {", ".join(missing_fields)}'
        }
    
    return {'valid': True, 'message': 'Data valid'}

def truncate_text(text: str, max_length: int = 100, add_ellipsis: bool = True) -> str:
    """Truncate text to specified length"""
    if not text or len(text) <= max_length:
        return text
    
    truncated = text[:max_length]
    
    if add_ellipsis:
        truncated += "..."
    
    return truncated

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely load JSON string"""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default

def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """Safely dump object to JSON string"""
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return default

def create_directory_if_not_exists(directory: str) -> bool:
    """Create directory if it doesn't exist"""
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False

def cleanup_temp_files(directory: str, max_age_hours: int = 24) -> int:
    """Clean up temporary files older than specified hours"""
    if not os.path.exists(directory):
        return 0
    
    current_time = datetime.now().timestamp()
    cutoff_time = current_time - (max_age_hours * 3600)
    
    cleaned_count = 0
    
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            if os.path.isfile(file_path):
                file_time = os.path.getmtime(file_path)
                
                if file_time < cutoff_time:
                    try:
                        os.remove(file_path)
                        cleaned_count += 1
                    except Exception:
                        continue
    
    except Exception:
        pass
    
    return cleaned_count

def encode_base64(data: bytes) -> str:
    """Encode bytes to base64 string"""
    try:
        return base64.b64encode(data).decode('utf-8')
    except Exception:
        return ""

def decode_base64(data: str) -> Optional[bytes]:
    """Decode base64 string to bytes"""
    try:
        return base64.b64decode(data)
    except Exception:
        return None

def sanitize_html(text: str) -> str:
    """Basic HTML sanitization"""
    if not text:
        return ""
    
    # Remove HTML tags
    clean_text = re.sub(r'<[^>]+>', '', text)
    
    # Decode HTML entities
    import html
    clean_text = html.unescape(clean_text)
    
    return clean_text

def format_timestamp(timestamp: str = None) -> str:
    """Format timestamp for display"""
    if timestamp:
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            return timestamp
    else:
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def calculate_confidence_score(scores: List[float]) -> float:
    """Calculate overall confidence from multiple scores"""
    if not scores:
        return 0.0
    
    # Weight higher scores more heavily
    weights = [score ** 2 for score in scores]
    weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
    weight_sum = sum(weights)
    
    if weight_sum == 0:
        return 0.0
    
    return min(1.0, weighted_sum / weight_sum)

def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalize score to specified range"""
    if max_val <= min_val:
        return min_val
    
    normalized = (score - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))

def merge_dictionaries(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries"""
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result

def filter_dict_by_keys(data: Dict[str, Any], allowed_keys: List[str]) -> Dict[str, Any]:
    """Filter dictionary to only include allowed keys"""
    return {k: v for k, v in data.items() if k in allowed_keys}

def get_nested_value(data: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get nested value from dictionary using dot notation"""
    keys = key_path.split('.')
    value = data
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default

def set_nested_value(data: Dict[str, Any], key_path: str, value: Any) -> None:
    """Set nested value in dictionary using dot notation"""
    keys = key_path.split('.')
    current = data
    
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value