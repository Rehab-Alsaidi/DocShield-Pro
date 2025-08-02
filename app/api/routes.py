# app/api/routes.py
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import uuid
import json
from datetime import datetime

from app.api.validators import validate_upload_request, validate_file
from utils.logger import get_logger
from utils.exceptions import ContentModerationError, ValidationError

logger = get_logger(__name__)

api_bp = Blueprint('api', __name__)

@api_bp.route('/upload', methods=['POST'])
def api_upload():
    """API endpoint for file upload and processing"""
    try:
        # Validate request
        validation_result = validate_upload_request(request)
        if not validation_result['valid']:
            return jsonify({
                'success': False,
                'error': validation_result['message']
            }), 400
        
        file = request.files['file']
        
        # Validate file
        file_validation = validate_file(file)
        if not file_validation['valid']:
            return jsonify({
                'success': False,
                'error': file_validation['message']
            }), 400
        
        # Save file
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        safe_filename = f"{file_id}_{filename}"
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], safe_filename)
        
        file.save(file_path)
        logger.info(f"API upload: {filename} -> {safe_filename}")
        
        # Get processing options
        options = {
            'include_images': request.form.get('include_images', 'true').lower() == 'true',
            'include_text': request.form.get('include_text', 'true').lower() == 'true',
            'confidence_threshold': float(request.form.get('confidence_threshold', 0.7))
        }
        
        # Process file
        result = current_app.content_moderator.moderate_pdf(file_path, filename)
        
        # Save results
        results_file = f"static/uploads/results_{file_id}.json"
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        # Return response
        return jsonify({
            'success': True,
            'result_id': file_id,
            'filename': filename,
            'processing_time': result.processing_metadata['processing_time_seconds'],
            'summary': {
                'total_violations': result.total_violations,
                'risk_level': result.overall_risk_level,
                'confidence': result.overall_confidence,
                'pages_analyzed': result.total_pages,
                'images_analyzed': result.total_images
            },
            'download_url': f'/api/results/{file_id}',
            'report_url': f'/api/report/{file_id}'
        })
        
    except ContentModerationError as e:
        logger.error(f"Content moderation error: {e}")
        return jsonify({
            'success': False,
            'error': f'Processing error: {str(e)}'
        }), 500
        
    except Exception as e:
        logger.error(f"API upload error: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@api_bp.route('/results/<result_id>', methods=['GET'])
def api_get_results(result_id):
    """Get processing results"""
    try:
        results_file = f"static/uploads/results_{result_id}.json"
        
        if not os.path.exists(results_file):
            return jsonify({
                'success': False,
                'error': 'Results not found'
            }), 404
        
        with open(results_file, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        
        # Filter response based on query parameters
        include_details = request.args.get('include_details', 'false').lower() == 'true'
        include_images = request.args.get('include_images', 'false').lower() == 'true'
        
        response_data = {
            'success': True,
            'result_id': result_id,
            'summary': {
                'document_id': result_data['document_id'],
                'file_name': result_data['file_name'],
                'processing_timestamp': result_data['processing_timestamp'],
                'total_violations': result_data['total_violations'],
                'overall_risk_level': result_data['overall_risk_level'],
                'overall_confidence': result_data['overall_confidence'],
                'total_pages': result_data['total_pages'],
                'total_images': result_data['total_images']
            }
        }
        
        if include_details:
            response_data['violations'] = result_data['violations']
            response_data['summary_stats'] = result_data['summary_stats']
            response_data['processing_metadata'] = result_data['processing_metadata']
        
        if include_images:
            # Include image data (base64 truncated for API)
            response_data['image_results'] = [
                {
                    'page_number': img['page_number'],
                    'caption': img['caption'],
                    'confidence': img['confidence'],
                    'size': img['size']
                } for img in result_data['image_results']
            ]
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"API get results error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve results'
        }), 500

@api_bp.route('/report/<result_id>', methods=['GET'])
def api_generate_report(result_id):
    """Generate and return PDF report"""
    try:
        results_file = f"static/uploads/results_{result_id}.json"
        
        if not os.path.exists(results_file):
            return jsonify({
                'success': False,
                'error': 'Results not found'
            }), 404
        
        with open(results_file, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        
        # Generate report
        report_path = current_app.report_generator.generate_pdf_report(result_data, result_id)
        
        return jsonify({
            'success': True,
            'report_url': f'/static/uploads/report_{result_id}.pdf',
            'download_url': f'/report/{result_id}'
        })
        
    except Exception as e:
        logger.error(f"API report generation error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to generate report'
        }), 500

@api_bp.route('/analyze', methods=['POST'])
def api_analyze_text():
    """Analyze text content directly"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Text content required'
            }), 400
        
        text = data['text']
        if len(text.strip()) < 10:
            return jsonify({
                'success': False,
                'error': 'Text too short for analysis'
            }), 400
        
        # Analyze text
        result = current_app.content_moderator.nlp_analyzer.analyze_text(text)
        
        return jsonify({
            'success': True,
            'analysis': {
                'risk_level': result.risk_level,
                'confidence': result.confidence,
                'total_words': result.total_words,
                'language': result.language,
                'sentiment_score': result.sentiment_score,
                'risk_keywords': len(result.risk_keywords),
                'content_categories': result.content_categories,
                'violations': [
                    {
                        'keyword': kw['keyword'],
                        'category': kw['category'],
                        'severity': kw['severity'],
                        'count': kw['count']
                    } for kw in result.risk_keywords
                ]
            }
        })
        
    except Exception as e:
        logger.error(f"API text analysis error: {e}")
        return jsonify({
            'success': False,
            'error': 'Analysis failed'
        }), 500

@api_bp.route('/health', methods=['GET'])
def api_health_check():
    """Health check endpoint"""
    try:
        # Check if models are loaded
        models_status = {
            'florence_model': hasattr(current_app.content_moderator.pdf_processor, 'florence_model'),
            'vision_analyzer': hasattr(current_app.content_moderator, 'vision_analyzer'),
            'nlp_analyzer': hasattr(current_app.content_moderator, 'nlp_analyzer')
        }
        
        all_models_loaded = all(models_status.values())
        
        return jsonify({
            'status': 'healthy' if all_models_loaded else 'degraded',
            'timestamp': datetime.now().isoformat(),
            'models': models_status,
            'upload_folder_exists': os.path.exists(current_app.config['UPLOAD_FOLDER']),
            'version': '1.0.0'
        })
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@api_bp.route('/stats', methods=['GET'])
def api_get_stats():
    """Get processing statistics"""
    try:
        # Count processed files
        upload_folder = current_app.config['UPLOAD_FOLDER']
        result_files = [f for f in os.listdir(upload_folder) if f.startswith('results_')]
        
        total_processed = len(result_files)
        
        # Aggregate statistics from recent results
        recent_stats = {
            'total_files_processed': total_processed,
            'recent_violations': 0,
            'avg_processing_time': 0,
            'risk_distribution': {'high': 0, 'medium': 0, 'low': 0}
        }
        
        # Process recent results (last 10)
        for result_file in result_files[-10:]:
            try:
                with open(os.path.join(upload_folder, result_file), 'r') as f:
                    data = json.load(f)
                    recent_stats['recent_violations'] += data.get('total_violations', 0)
                    recent_stats['avg_processing_time'] += data.get('processing_metadata', {}).get('processing_time_seconds', 0)
                    
                    risk_level = data.get('overall_risk_level', 'low')
                    recent_stats['risk_distribution'][risk_level] += 1
                    
            except Exception:
                continue
        
        if len(result_files) > 0:
            recent_stats['avg_processing_time'] /= min(10, len(result_files))
        
        return jsonify({
            'success': True,
            'statistics': recent_stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve statistics'
        }), 500

@api_bp.errorhandler(413)
def api_file_too_large(e):
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 50MB.'
    }), 413

@api_bp.errorhandler(400)
def api_bad_request(e):
    return jsonify({
        'success': False,
        'error': 'Bad request'
    }), 400

@api_bp.errorhandler(500)
def api_internal_error(e):
    logger.error(f"API internal error: {e}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500