# services/file_manager.py
import os
import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

from utils.logger import get_logger
from utils.helpers import get_file_info, format_file_size, create_unique_filename
from utils.exceptions import FileUploadError, FileSizeError, FileTypeError, ResourceNotFoundError
from app.config import app_config

logger = get_logger(__name__)

class FileManager:
    """Manages file operations for the content moderation system"""
    
    def __init__(self):
        self.upload_folder = Path(app_config.upload_folder)
        self.temp_folder = Path("temp")
        self.results_folder = Path("results")
        self.reports_folder = Path("reports")
        
        # Create directories if they don't exist
        self._ensure_directories()
        
        logger.info("File Manager initialized")
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.upload_folder,
            self.temp_folder,
            self.results_folder,
            self.reports_folder
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
    
    def save_uploaded_file(self, file_storage, original_filename: str) -> Tuple[str, str]:
        """Save uploaded file and return file ID and file path"""
        try:
            # Generate unique file ID
            file_id = str(uuid.uuid4())
            
            # Create secure filename
            secure_name = create_unique_filename(original_filename)
            filename_with_id = f"{file_id}_{secure_name}"
            
            # Full file path
            file_path = self.upload_folder / filename_with_id
            
            # Save file
            file_storage.save(str(file_path))
            
            # Verify file was saved correctly
            if not file_path.exists():
                raise FileUploadError(f"Failed to save file: {original_filename}")
            
            # Get file info
            file_info = get_file_info(str(file_path))
            
            # Log upload
            logger.info(f"File uploaded successfully: {original_filename} -> {filename_with_id}")
            logger.info(f"File size: {format_file_size(file_info['size_bytes'])}")
            
            return file_id, str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save uploaded file: {e}")
            raise FileUploadError(f"Failed to save file: {str(e)}", filename=original_filename)
    
    def save_processing_results(self, results: Dict, file_id: str) -> str:
        """Save processing results to JSON file"""
        try:
            results_filename = f"results_{file_id}.json"
            results_path = self.results_folder / results_filename
            
            # Add metadata
            results_with_metadata = {
                **results,
                'file_metadata': {
                    'file_id': file_id,
                    'saved_timestamp': datetime.now().isoformat(),
                    'results_version': '1.0'
                }
            }
            
            # Save results
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results_with_metadata, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Results saved: {results_filename}")
            return str(results_path)
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise FileUploadError(f"Failed to save results: {str(e)}")
    
    def load_processing_results(self, file_id: str) -> Dict:
        """Load processing results from JSON file"""
        try:
            results_filename = f"results_{file_id}.json"
            results_path = self.results_folder / results_filename
            
            if not results_path.exists():
                raise ResourceNotFoundError(
                    f"Results not found for file ID: {file_id}",
                    resource_type="processing_results",
                    resource_id=file_id
                )
            
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            logger.debug(f"Results loaded: {results_filename}")
            return results
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse results JSON: {e}")
            raise FileUploadError(f"Corrupted results file for ID: {file_id}")
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            raise
    
    def save_report(self, report_data: bytes, file_id: str, report_type: str = "pdf") -> str:
        """Save generated report"""
        try:
            report_filename = f"report_{file_id}.{report_type}"
            report_path = self.reports_folder / report_filename
            
            with open(report_path, 'wb') as f:
                f.write(report_data)
            
            logger.info(f"Report saved: {report_filename}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            raise FileUploadError(f"Failed to save report: {str(e)}")
    
    def get_file_path(self, file_id: str) -> Optional[str]:
        """Get file path for a given file ID"""
        try:
            # Search for file with this ID
            for file_path in self.upload_folder.glob(f"{file_id}_*"):
                if file_path.is_file():
                    return str(file_path)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to find file path for ID {file_id}: {e}")
            return None
    
    def delete_file(self, file_id: str) -> bool:
        """Delete uploaded file"""
        try:
            file_path = self.get_file_path(file_id)
            
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"File deleted: {file_id}")
                return True
            else:
                logger.warning(f"File not found for deletion: {file_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
            return False
    
    def delete_results(self, file_id: str) -> bool:
        """Delete processing results"""
        try:
            results_path = self.results_folder / f"results_{file_id}.json"
            
            if results_path.exists():
                results_path.unlink()
                logger.info(f"Results deleted: {file_id}")
                return True
            else:
                logger.warning(f"Results not found for deletion: {file_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete results {file_id}: {e}")
            return False
    
    def delete_report(self, file_id: str, report_type: str = "pdf") -> bool:
        """Delete generated report"""
        try:
            report_path = self.reports_folder / f"report_{file_id}.{report_type}"
            
            if report_path.exists():
                report_path.unlink()
                logger.info(f"Report deleted: {file_id}")
                return True
            else:
                logger.warning(f"Report not found for deletion: {file_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete report {file_id}: {e}")
            return False
    
    def cleanup_old_files(self, max_age_hours: int = 24) -> Dict[str, int]:
        """Clean up old files and results"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleanup_stats = {
            'uploaded_files': 0,
            'results_files': 0,
            'report_files': 0,
            'temp_files': 0
        }
        
        try:
            # Clean uploaded files
            cleanup_stats['uploaded_files'] = self._cleanup_directory(
                self.upload_folder, cutoff_time
            )
            
            # Clean results files
            cleanup_stats['results_files'] = self._cleanup_directory(
                self.results_folder, cutoff_time
            )
            
            # Clean report files
            cleanup_stats['report_files'] = self._cleanup_directory(
                self.reports_folder, cutoff_time
            )
            
            # Clean temp files
            cleanup_stats['temp_files'] = self._cleanup_directory(
                self.temp_folder, cutoff_time
            )
            
            total_cleaned = sum(cleanup_stats.values())
            logger.info(f"Cleanup completed: {total_cleaned} files removed")
            
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return cleanup_stats
    
    def _cleanup_directory(self, directory: Path, cutoff_time: datetime) -> int:
        """Clean up files in a specific directory"""
        cleaned_count = 0
        
        try:
            for file_path in directory.iterdir():
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    if file_time < cutoff_time:
                        try:
                            file_path.unlink()
                            cleaned_count += 1
                            logger.debug(f"Cleaned up file: {file_path.name}")
                        except Exception as e:
                            logger.warning(f"Failed to delete {file_path.name}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup directory {directory}: {e}")
        
        return cleaned_count
    
    def get_storage_stats(self) -> Dict[str, any]:
        """Get storage statistics"""
        try:
            stats = {}
            
            for folder_name, folder_path in [
                ('uploads', self.upload_folder),
                ('results', self.results_folder),
                ('reports', self.reports_folder),
                ('temp', self.temp_folder)
            ]:
                if folder_path.exists():
                    file_count = len(list(folder_path.glob('*')))
                    total_size = sum(f.stat().st_size for f in folder_path.glob('*') if f.is_file())
                    
                    stats[folder_name] = {
                        'file_count': file_count,
                        'total_size_bytes': total_size,
                        'total_size_formatted': format_file_size(total_size)
                    }
                else:
                    stats[folder_name] = {
                        'file_count': 0,
                        'total_size_bytes': 0,
                        'total_size_formatted': '0 B'
                    }
            
            # Calculate total
            total_files = sum(stat['file_count'] for stat in stats.values())
            total_size = sum(stat['total_size_bytes'] for stat in stats.values())
            
            stats['total'] = {
                'file_count': total_files,
                'total_size_bytes': total_size,
                'total_size_formatted': format_file_size(total_size)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}
    
    def list_files(self, folder_type: str = 'uploads', limit: int = 100) -> List[Dict]:
        """List files in a specific folder"""
        try:
            folder_map = {
                'uploads': self.upload_folder,
                'results': self.results_folder,
                'reports': self.reports_folder,
                'temp': self.temp_folder
            }
            
            if folder_type not in folder_map:
                raise ValueError(f"Invalid folder type: {folder_type}")
            
            folder_path = folder_map[folder_type]
            files = []
            
            for file_path in sorted(folder_path.glob('*'), key=lambda x: x.stat().st_mtime, reverse=True):
                if file_path.is_file():
                    file_info = get_file_info(str(file_path))
                    files.append({
                        'filename': file_path.name,
                        'size': file_info['size_formatted'] if 'size_formatted' in file_info else format_file_size(file_info['size_bytes']),
                        'modified': file_info['modified'],
                        'mime_type': file_info['mime_type']
                    })
                    
                    if len(files) >= limit:
                        break
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    def create_temp_file(self, content: bytes, extension: str = "tmp") -> str:
        """Create temporary file with content"""
        try:
            temp_id = str(uuid.uuid4())
            temp_filename = f"temp_{temp_id}.{extension}"
            temp_path = self.temp_folder / temp_filename
            
            with open(temp_path, 'wb') as f:
                f.write(content)
            
            logger.debug(f"Temporary file created: {temp_filename}")
            return str(temp_path)
            
        except Exception as e:
            logger.error(f"Failed to create temp file: {e}")
            raise FileUploadError(f"Failed to create temporary file: {str(e)}")
    
    def validate_file_upload(self, file_storage, max_size: int = None) -> Dict[str, any]:
        """Validate uploaded file"""
        max_size = max_size or app_config.max_file_size
        
        # Check file size
        if hasattr(file_storage, 'content_length') and file_storage.content_length:
            if file_storage.content_length > max_size:
                raise FileSizeError(
                    f"File too large. Maximum size is {format_file_size(max_size)}",
                    filename=file_storage.filename,
                    actual_size=file_storage.content_length,
                    max_size=max_size
                )
        
        # Check file type
        if not file_storage.filename.lower().endswith('.pdf'):
            raise FileTypeError(
                "Only PDF files are allowed",
                filename=file_storage.filename,
                actual_type=file_storage.mimetype,
                allowed_types=['application/pdf']
            )
        
        # Verify PDF header
        try:
            file_storage.seek(0)
            header = file_storage.read(8)
            file_storage.seek(0)
            
            if not header.startswith(b'%PDF-'):
                raise FileTypeError(
                    "File does not appear to be a valid PDF",
                    filename=file_storage.filename
                )
        except Exception as e:
            raise FileUploadError(f"Unable to validate file: {str(e)}")
        
        return {
            'valid': True,
            'filename': file_storage.filename,
            'size': getattr(file_storage, 'content_length', 0),
            'mime_type': file_storage.mimetype
        }