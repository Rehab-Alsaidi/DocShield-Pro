# app/enhanced_routes.py
"""
Enhanced Routes for Large File Processing and Professional System
Handles 4GB+ files, real-time progress, and comprehensive reporting
"""

import os
import json
import uuid
import asyncio
import threading
from datetime import datetime
from flask import Blueprint, request, render_template, jsonify, send_file, session, redirect, url_for
from werkzeug.utils import secure_filename
import logging

from utils.logger import get_logger
from core.large_file_processor import LargeFilePDFProcessor, ProcessingProgress

logger = get_logger(__name__)

try:
    from core.enhanced_vision_analyzer import EnhancedVisionAnalyzer
except ImportError:
    from core.simple_vision_analyzer import SimpleVisionAnalyzer as EnhancedVisionAnalyzer
    logger.warning("Using SimpleVisionAnalyzer as fallback for EnhancedVisionAnalyzer")
try:
    from core.image_generator import CulturalImageGenerator
except ImportError:
    from core.working_image_generator import WorkingImageGenerator as CulturalImageGenerator
    logger.warning("Using WorkingImageGenerator as fallback for CulturalImageGenerator")
try:
    from services.enhanced_report_generator import EnhancedReportGenerator
except ImportError as e:
    logger.warning(f"Enhanced report generator not available: {e}")
    # Create a simple fallback
    class EnhancedReportGenerator:
        def __init__(self):
            pass
        def generate_pdf_report(self, *args, **kwargs):
            raise NotImplementedError("Enhanced report generator not available")

# Create blueprint
enhanced_bp = Blueprint('enhanced', __name__)

# Global processing status tracking
processing_status = {}
processing_locks = {}

class ProcessingManager:
    """Manage processing tasks and progress tracking"""
    
    def __init__(self):
        self.active_tasks = {}
        self.completed_tasks = {}
        
    def start_processing(self, task_id: str, pdf_path: str, options: dict):
        """Start processing task in background"""
        
        def processing_worker():
            try:
                logger.info(f"Starting background processing for task {task_id}")
                
                # Initialize processors
                processor = LargeFilePDFProcessor(
                    max_memory_mb=options.get('max_memory_mb', 2048),
                    chunk_size=options.get('chunk_size', 10)
                )
                
                vision_analyzer = EnhancedVisionAnalyzer()
                image_generator = CulturalImageGenerator()
                report_generator = EnhancedReportGenerator()
                
                # Set up progress callback
                def progress_callback(progress: ProcessingProgress):
                    processing_status[task_id] = {
                        'status': 'processing',
                        'progress': progress,
                        'timestamp': datetime.now().isoformat()
                    }
                
                processor.set_progress_callback(progress_callback)
                
                # Process the file
                processed_images, text_data, final_progress = processor.process_large_pdf(pdf_path)
                
                # Enhanced analysis
                enhanced_results = []
                for img_result in processed_images:
                    # Convert base64 back to PIL for analysis
                    from PIL import Image
                    import base64
                    import io
                    
                    try:
                        # Decode base64 image
                        image_data = img_result.image_base64.split(',')[1]
                        image_bytes = base64.b64decode(image_data)
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        
                        # Enhanced vision analysis
                        vision_result = vision_analyzer.analyze_image_comprehensive(
                            pil_image, img_result.caption
                        )
                        
                        # Add enhanced data to result
                        img_result.enhanced_analysis = vision_result
                        enhanced_results.append(img_result)
                        
                        # Generate replacement if needed
                        if vision_result.risk_level in ['high', 'critical']:
                            try:
                                replacement_suggestion = vision_result.replacement_suggestions[0]
                                original_analysis = {
                                    "concept": img_result.caption,
                                    "category": vision_result.detected_categories,
                                    "severity": vision_result.risk_level
                                }
                                
                                generated_image = image_generator.generate_replacement_image(
                                    original_analysis, replacement_suggestion
                                )
                                
                                img_result.replacement_image = generated_image
                                
                            except Exception as e:
                                logger.warning(f"Failed to generate replacement: {e}")
                        
                    except Exception as e:
                        logger.error(f"Enhanced analysis failed for image: {e}")
                        enhanced_results.append(img_result)
                
                # Create comprehensive result
                from core.content_moderator import ModerationResult
                
                result = ModerationResult(
                    document_id=task_id,
                    file_name=os.path.basename(pdf_path),
                    processing_timestamp=datetime.now().isoformat(),
                    total_pages=final_progress.total_pages,
                    total_images=len(enhanced_results),
                    total_violations=sum(len(getattr(img, 'violations', [])) for img in enhanced_results),
                    overall_risk_level=self._calculate_overall_risk(enhanced_results),
                    overall_confidence=0.85,
                    image_results=enhanced_results,
                    text_results=text_data,
                    vision_results=[getattr(img, 'enhanced_analysis', None) for img in enhanced_results],
                    violations=self._extract_all_violations(enhanced_results, text_data),
                    summary_stats=self._generate_summary_stats(enhanced_results, text_data),
                    processing_metadata={
                        'processing_time_seconds': final_progress.estimated_completion - final_progress.start_time,
                        'pdf_file_size_mb': round(os.path.getsize(pdf_path) / 1024 / 1024, 2),
                        'models_used': ['Enhanced Vision Analyzer', 'Cultural Image Generator', 'Large File Processor'],
                        'chunk_size': options.get('chunk_size', 10),
                        'max_memory_mb': options.get('max_memory_mb', 2048)
                    }
                )
                
                # Generate reports
                html_report = report_generator.generate_comprehensive_report(result, 'html')
                pdf_report = report_generator.generate_comprehensive_report(result, 'pdf')
                json_report = report_generator.generate_comprehensive_report(result, 'json')
                
                # Store completed task
                self.completed_tasks[task_id] = {
                    'result': result,
                    'reports': {
                        'html': html_report,
                        'pdf': pdf_report,
                        'json': json_report
                    },
                    'completion_time': datetime.now().isoformat()
                }
                
                # Update status
                processing_status[task_id] = {
                    'status': 'completed',
                    'progress': final_progress,
                    'result': result,
                    'reports': {
                        'html': html_report,
                        'pdf': pdf_report,
                        'json': json_report
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"Processing completed for task {task_id}")
                
            except Exception as e:
                logger.error(f"Processing failed for task {task_id}: {e}")
                processing_status[task_id] = {
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        # Start processing in background thread
        thread = threading.Thread(target=processing_worker, daemon=True)
        thread.start()
        
        # Store task info
        self.active_tasks[task_id] = {
            'thread': thread,
            'start_time': datetime.now(),
            'pdf_path': pdf_path,
            'options': options
        }
    
    def _calculate_overall_risk(self, enhanced_results) -> str:
        """Calculate overall risk level from enhanced results"""
        risk_scores = []
        
        for result in enhanced_results:
            if hasattr(result, 'enhanced_analysis') and result.enhanced_analysis:
                if result.enhanced_analysis.risk_level == 'critical':
                    risk_scores.append(4)
                elif result.enhanced_analysis.risk_level == 'high':
                    risk_scores.append(3)
                elif result.enhanced_analysis.risk_level == 'medium':
                    risk_scores.append(2)
                else:
                    risk_scores.append(1)
        
        if not risk_scores:
            return 'low'
        
        avg_risk = sum(risk_scores) / len(risk_scores)
        
        if avg_risk >= 3.5:
            return 'critical'
        elif avg_risk >= 2.5:
            return 'high'
        elif avg_risk >= 1.5:
            return 'medium'
        else:
            return 'low'
    
    def _extract_all_violations(self, enhanced_results, text_data):
        """Extract all violations from enhanced results"""
        violations = []
        
        # Extract image violations
        for result in enhanced_results:
            if hasattr(result, 'violations'):
                violations.extend(result.violations)
        
        # Extract text violations (if any)
        for text_result in text_data:
            if 'violations' in text_result:
                violations.extend(text_result['violations'])
        
        return violations
    
    def _generate_summary_stats(self, enhanced_results, text_data):
        """Generate summary statistics"""
        return {
            'total_images_analyzed': len(enhanced_results),
            'images_with_replacements': len([r for r in enhanced_results if hasattr(r, 'replacement_image')]),
            'text_blocks_analyzed': len(text_data),
            'total_highlighted_words': sum(td.get('highlighted_count', 0) for td in text_data),
            'cultural_compliance': self._calculate_cultural_compliance(enhanced_results)
        }
    
    def _calculate_cultural_compliance(self, enhanced_results):
        """Calculate cultural compliance score"""
        compliance_scores = []
        
        for result in enhanced_results:
            if hasattr(result, 'enhanced_analysis') and result.enhanced_analysis:
                compliance_scores.append(result.enhanced_analysis.cultural_compliance_score)
        
        return sum(compliance_scores) / len(compliance_scores) if compliance_scores else 0.7

# Initialize processing manager
processing_manager = ProcessingManager()

@enhanced_bp.route('/enhanced_upload')
def enhanced_upload():
    """Enhanced upload page for large files"""
    return render_template('upload_new.html')

@enhanced_bp.route('/process_large_file', methods=['POST'])
def process_large_file():
    """Process large PDF file with enhanced features"""
    
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are supported'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        task_id = str(uuid.uuid4())
        upload_path = os.path.join('static/uploads', f"{task_id}_{filename}")
        
        os.makedirs(os.path.dirname(upload_path), exist_ok=True)
        file.save(upload_path)
        
        # Get processing options
        options = {
            'max_memory_mb': int(request.form.get('max_memory_mb', 2048)),
            'chunk_size': int(request.form.get('chunk_size', 10)),
            'include_replacements': request.form.get('include_replacements', 'true').lower() == 'true',
            'cultural_focus': request.form.get('cultural_focus', 'islamic_middle_eastern')
        }
        
        # Start background processing
        processing_manager.start_processing(task_id, upload_path, options)
        
        # Initialize status
        processing_status[task_id] = {
            'status': 'started',
            'file_name': filename,
            'task_id': task_id,
            'options': options,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Started processing task {task_id} for file {filename}")
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': 'Processing started',
            'redirect_url': url_for('enhanced.processing_status', task_id=task_id)
        })
        
    except Exception as e:
        logger.error(f"File processing initiation failed: {e}")
        return jsonify({'error': str(e)}), 500

@enhanced_bp.route('/processing_status/<task_id>')
def processing_status_page(task_id):
    """Show processing status page with real-time updates"""
    
    if task_id not in processing_status:
        return render_template('error.html', 
                             error="Processing task not found"), 404
    
    return render_template('processing_status.html', 
                         task_id=task_id,
                         initial_status=processing_status[task_id])

@enhanced_bp.route('/api/status/<task_id>')
def get_processing_status(task_id):
    """API endpoint for real-time processing status"""
    
    if task_id not in processing_status:
        return jsonify({'error': 'Task not found'}), 404
    
    status = processing_status[task_id]
    
    # Format progress data for JSON response
    response_data = {
        'task_id': task_id,
        'status': status['status'],
        'timestamp': status['timestamp']
    }
    
    if 'progress' in status:
        progress = status['progress']
        response_data['progress'] = {
            'percentage': getattr(progress, 'progress_percentage', 0),
            'current_phase': getattr(progress, 'current_phase', 'Unknown'),
            'processed_pages': getattr(progress, 'processed_pages', 0),
            'total_pages': getattr(progress, 'total_pages', 0),
            'processed_images': getattr(progress, 'processed_images', 0),
            'memory_usage_mb': getattr(progress, 'memory_usage_mb', 0)
        }
    
    if 'error' in status:
        response_data['error'] = status['error']
    
    if 'result' in status:
        # Include basic result info (not full result to avoid large responses)
        result = status['result']
        response_data['result_summary'] = {
            'total_violations': result.total_violations,
            'risk_level': result.overall_risk_level,
            'total_images': result.total_images,
            'confidence': result.overall_confidence
        }
    
    if 'reports' in status:
        response_data['reports_ready'] = True
        response_data['report_urls'] = {
            'preview': url_for('enhanced.results_preview', task_id=task_id),
            'download_html': url_for('enhanced.download_report', task_id=task_id, format='html'),
            'download_pdf': url_for('enhanced.download_report', task_id=task_id, format='pdf'),
            'download_json': url_for('enhanced.download_report', task_id=task_id, format='json')
        }
    
    return jsonify(response_data)

@enhanced_bp.route('/results_preview/<task_id>')
def results_preview(task_id):
    """Show detailed results preview before download"""
    
    if task_id not in processing_status:
        return render_template('error.html', 
                             error="Results not found"), 404
    
    status = processing_status[task_id]
    
    if status['status'] != 'completed':
        return redirect(url_for('enhanced.processing_status_page', task_id=task_id))
    
    result = status['result']
    
    return render_template('results.html', 
                         result=result,
                         task_id=task_id,
                         reports=status.get('reports', {}))

@enhanced_bp.route('/download_report/<task_id>/<format>')
def download_report(task_id, format):
    """Download generated report in specified format"""
    
    if task_id not in processing_status:
        return jsonify({'error': 'Task not found'}), 404
    
    status = processing_status[task_id]
    
    if status['status'] != 'completed' or 'reports' not in status:
        return jsonify({'error': 'Reports not ready'}), 404
    
    reports = status['reports']
    
    if format not in reports:
        return jsonify({'error': f'Format {format} not available'}), 404
    
    report_path = reports[format]
    
    if not os.path.exists(report_path):
        return jsonify({'error': 'Report file not found'}), 404
    
    # Determine MIME type
    mime_types = {
        'html': 'text/html',
        'pdf': 'application/pdf',
        'json': 'application/json'
    }
    
    return send_file(
        report_path,
        as_attachment=True,
        download_name=f"content_analysis_{task_id}.{format}",
        mimetype=mime_types.get(format, 'application/octet-stream')
    )

@enhanced_bp.route('/file_info', methods=['POST'])
def get_file_info():
    """Get information about uploaded file before processing"""
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Save temporarily to analyze
        temp_filename = secure_filename(file.filename)
        temp_path = os.path.join('temp', f"temp_{temp_filename}")
        
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        file.save(temp_path)
        
        # Analyze file
        processor = LargeFilePDFProcessor()
        file_info = processor.get_file_info(temp_path)
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
        
        return jsonify(file_info)
        
    except Exception as e:
        logger.error(f"File info analysis failed: {e}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@enhanced_bp.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return render_template('error.html', 
                         error="File too large. Maximum size is 4GB."), 413

@enhanced_bp.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {error}")
    return render_template('error.html', 
                         error="Internal server error occurred."), 500