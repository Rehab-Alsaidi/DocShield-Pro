# services/enhanced_report_generator.py
"""
Enhanced Report Generator with Before/After Comparisons
Generates comprehensive reports with visual comparisons and detailed analysis
"""

import os
import json
import base64
from datetime import datetime
from typing import Dict, List, Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML, CSS
from PIL import Image, ImageDraw, ImageFont
import io
import logging

from core.image_generator import CulturalImageGenerator, GeneratedImage
from utils.logger import get_logger

logger = get_logger(__name__)

class EnhancedReportGenerator:
    """Generate comprehensive reports with before/after image comparisons"""
    
    def __init__(self, template_dir: str = "templates", output_dir: str = "reports"):
        self.template_dir = template_dir
        self.output_dir = output_dir
        self.image_generator = CulturalImageGenerator()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )
        
        # Add custom filters
        self._add_custom_filters()
        
        logger.info(f"Enhanced Report Generator initialized")
    
    def _add_custom_filters(self):
        """Add custom Jinja2 filters for report generation"""
        
        def highlight_keywords(text: str, risk_keywords: List[Dict]) -> str:
            """Highlight risk keywords in text"""
            if not risk_keywords or not text:
                return text
            
            highlighted = text
            for keyword_info in risk_keywords:
                keyword = keyword_info.get('keyword', '')
                severity = keyword_info.get('severity', 'medium')
                
                # Create highlight span based on severity
                if severity == 'high':
                    highlight_class = 'highlight-high'
                elif severity == 'medium':
                    highlight_class = 'highlight-medium'
                else:
                    highlight_class = 'highlight-low'
                
                highlighted = highlighted.replace(
                    keyword,
                    f'<span class="{highlight_class}">{keyword}</span>'
                )
            
            return highlighted
        
        def format_confidence(confidence: float) -> str:
            """Format confidence as percentage"""
            return f"{confidence * 100:.1f}%"
        
        def severity_color(severity: str) -> str:
            """Get color class for severity"""
            colors = {
                'critical': 'danger',
                'high': 'warning', 
                'medium': 'info',
                'low': 'secondary'
            }
            return colors.get(severity.lower(), 'secondary')
        
        def strftime(timestamp, fmt='%Y-%m-%d %H:%M:%S'):
            """Format timestamp"""
            if isinstance(timestamp, str):
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    return dt.strftime(fmt)
                except:
                    return timestamp
            return timestamp
        
        # Register filters
        self.jinja_env.filters['highlight_keywords'] = highlight_keywords
        self.jinja_env.filters['format_confidence'] = format_confidence
        self.jinja_env.filters['severity_color'] = severity_color
        self.jinja_env.filters['strftime'] = strftime
    
    def generate_comprehensive_report(self, moderation_result, 
                                    format_type: str = 'html',
                                    include_replacements: bool = True) -> str:
        """Generate comprehensive report with all enhancements"""
        
        logger.info(f"Generating comprehensive {format_type} report for {moderation_result.file_name}")
        
        try:
            # Enhance results with replacement images if requested
            if include_replacements:
                moderation_result = self._add_replacement_images(moderation_result)
            
            # Generate report based on format
            if format_type.lower() == 'html':
                return self._generate_html_report(moderation_result)
            elif format_type.lower() == 'pdf':
                return self._generate_pdf_report(moderation_result)
            elif format_type.lower() == 'json':
                return self._generate_json_report(moderation_result)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise
    
    def _add_replacement_images(self, moderation_result):
        """Add replacement images to violation results"""
        
        logger.info("Generating replacement images for violations...")
        
        # Process image violations that need replacements
        for image_result in moderation_result.image_results:
            if hasattr(image_result, 'violations') and image_result.violations:
                
                # Find high-severity violations that need replacements
                high_severity_violations = [
                    v for v in image_result.violations 
                    if v.severity in ['high', 'critical']
                ]
                
                if high_severity_violations:
                    try:
                        # Generate replacement suggestion
                        replacement_suggestion = self._create_replacement_suggestion(
                            high_severity_violations[0]  # Use first high-severity violation
                        )
                        
                        # Generate replacement image
                        original_analysis = {
                            "concept": image_result.caption,
                            "category": high_severity_violations[0].category,
                            "severity": high_severity_violations[0].severity
                        }
                        
                        generated_image = self.image_generator.generate_replacement_image(
                            original_analysis, replacement_suggestion
                        )
                        
                        # Add replacement to image result
                        image_result.replacement_image = generated_image
                        
                        # Create before/after comparison
                        original_image = self._base64_to_pil(image_result.image_base64)
                        if original_image:
                            comparison_image = self.image_generator.generate_before_after_comparison(
                                original_image, generated_image
                            )
                            
                            # Convert comparison to base64
                            comparison_base64 = self._pil_to_base64(comparison_image)
                            image_result.comparison_image = comparison_base64
                        
                        logger.info(f"Generated replacement for page {image_result.page_number}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to generate replacement for page {image_result.page_number}: {e}")
        
        return moderation_result
    
    def _create_replacement_suggestion(self, violation) -> Dict:
        """Create replacement suggestion based on violation"""
        
        suggestion_mapping = {
            "adult_content": {
                "replacement_type": "general_appropriate",
                "generation_prompt": "Professional appropriate content suitable for all audiences",
                "priority": "critical"
            },
            "alcohol": {
                "replacement_type": "beverage_alternative", 
                "generation_prompt": "Traditional Arabic coffee or tea service in elegant setting",
                "priority": "high"
            },
            "gambling": {
                "replacement_type": "activity_replacement",
                "generation_prompt": "Professional educational seminar or business meeting",
                "priority": "high"
            },
            "revealing_clothing": {
                "replacement_type": "clothing_replacement",
                "generation_prompt": "Professional business attire with modest dress code",
                "priority": "high"
            },
            "inappropriate_content": {
                "replacement_type": "general_appropriate",
                "generation_prompt": "Culturally appropriate professional content",
                "priority": "medium"
            }
        }
        
        # Default suggestion
        default_suggestion = {
            "replacement_type": "general_appropriate",
            "generation_prompt": "Professional, culturally appropriate content",
            "priority": "medium"
        }
        
        return suggestion_mapping.get(violation.category, default_suggestion)
    
    def _generate_html_report(self, moderation_result) -> str:
        """Generate comprehensive HTML report"""
        
        try:
            # Load template
            template = self.jinja_env.get_template('comprehensive_report.html')
            
            # Prepare enhanced data
            report_data = {
                'result': moderation_result,
                'generation_timestamp': datetime.now().isoformat(),
                'report_metadata': {
                    'version': '2.0',
                    'type': 'comprehensive_analysis',
                    'includes_replacements': True,
                    'cultural_compliance_focus': 'Islamic/Middle Eastern'
                },
                'summary': self._generate_executive_summary(moderation_result),
                'recommendations': self._generate_recommendations(moderation_result)
            }
            
            # Render template
            html_content = template.render(**report_data)
            
            # Save HTML report
            output_filename = f"comprehensive_report_{moderation_result.document_id}.html"
            output_path = os.path.join(self.output_dir, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML report saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"HTML report generation failed: {e}")
            raise
    
    def _generate_pdf_report(self, moderation_result) -> str:
        """Generate PDF report from HTML template"""
        
        try:
            # First generate HTML content
            template = self.jinja_env.get_template('comprehensive_report.html')
            
            report_data = {
                'result': moderation_result,
                'generation_timestamp': datetime.now().isoformat(),
                'is_pdf_version': True,
                'summary': self._generate_executive_summary(moderation_result),
                'recommendations': self._generate_recommendations(moderation_result)
            }
            
            html_content = template.render(**report_data)
            
            # Convert to PDF
            output_filename = f"comprehensive_report_{moderation_result.document_id}.pdf"
            output_path = os.path.join(self.output_dir, output_filename)
            
            # Custom CSS for PDF
            pdf_css = CSS(string='''
                @page {
                    size: A4;
                    margin: 2cm;
                }
                
                body {
                    font-family: Arial, sans-serif;
                    font-size: 12px;
                    line-height: 1.4;
                }
                
                .page-break {
                    page-break-before: always;
                }
                
                .no-break {
                    page-break-inside: avoid;
                }
                
                img {
                    max-width: 100%;
                    height: auto;
                }
                
                .comparison-images {
                    display: flex;
                    justify-content: space-between;
                    margin: 1rem 0;
                }
                
                .comparison-images img {
                    width: 45%;
                }
            ''')
            
            HTML(string=html_content).write_pdf(output_path, stylesheets=[pdf_css])
            
            logger.info(f"PDF report saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"PDF report generation failed: {e}")
            # Fallback to HTML if PDF fails
            logger.info("Falling back to HTML report...")
            return self._generate_html_report(moderation_result)
    
    def _generate_json_report(self, moderation_result) -> str:
        """Generate JSON data export"""
        
        try:
            # Convert moderation result to dictionary
            report_dict = self._result_to_dict(moderation_result)
            
            # Add metadata
            report_dict['export_metadata'] = {
                'export_timestamp': datetime.now().isoformat(),
                'export_version': '2.0',
                'includes_base64_images': True,
                'includes_replacements': True
            }
            
            # Save JSON
            output_filename = f"analysis_data_{moderation_result.document_id}.json"
            output_path = os.path.join(self.output_dir, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"JSON report saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"JSON report generation failed: {e}")
            raise
    
    def _generate_executive_summary(self, moderation_result) -> Dict:
        """Generate executive summary for the report"""
        
        violations = moderation_result.violations
        
        # Risk distribution
        risk_counts = {
            'critical': len([v for v in violations if v.severity == 'critical']),
            'high': len([v for v in violations if v.severity == 'high']),
            'medium': len([v for v in violations if v.severity == 'medium']),
            'low': len([v for v in violations if v.severity == 'low'])
        }
        
        # Category analysis
        from collections import Counter
        category_counts = Counter([v.category for v in violations])
        top_categories = category_counts.most_common(5)
        
        # Cultural compliance assessment
        cultural_violations = [
            v for v in violations 
            if v.category in ['adult_content', 'alcohol', 'gambling', 'revealing_clothing', 'inappropriate_content']
        ]
        
        cultural_compliance_issues = len(cultural_violations) > 0
        
        summary = {
            'overall_assessment': {
                'risk_level': moderation_result.overall_risk_level,
                'confidence': moderation_result.overall_confidence,
                'total_violations': len(violations),
                'cultural_compliance_issues': cultural_compliance_issues
            },
            'risk_distribution': risk_counts,
            'top_violation_categories': top_categories,
            'processing_stats': {
                'pages_analyzed': moderation_result.total_pages,
                'images_analyzed': moderation_result.total_images,
                'processing_time': moderation_result.processing_metadata.get('processing_time_seconds', 0)
            },
            'key_findings': self._generate_key_findings(moderation_result)
        }
        
        return summary
    
    def _generate_key_findings(self, moderation_result) -> List[str]:
        """Generate key findings list"""
        
        findings = []
        violations = moderation_result.violations
        
        # High-level findings
        if moderation_result.overall_risk_level == 'critical':
            findings.append("Document contains critical content violations requiring immediate attention")
        elif moderation_result.overall_risk_level == 'high':
            findings.append("Document has significant content issues that need review")
        
        # Category-specific findings
        category_counts = {}
        for violation in violations:
            category_counts[violation.category] = category_counts.get(violation.category, 0) + 1
        
        if 'adult_content' in category_counts:
            findings.append(f"Adult content detected in {category_counts['adult_content']} instances")
        
        if 'alcohol' in category_counts:
            findings.append(f"Alcohol-related content found in {category_counts['alcohol']} instances")
        
        if 'gambling' in category_counts:
            findings.append(f"Gambling content identified in {category_counts['gambling']} instances")
        
        # Positive findings
        if not violations:
            findings.append("No content violations detected - document appears compliant")
        elif len(violations) < 3:
            findings.append("Minimal content issues detected - overall good compliance")
        
        # Processing findings
        if moderation_result.total_images > 10:
            findings.append(f"Comprehensive analysis performed on {moderation_result.total_images} images")
        
        return findings
    
    def _generate_recommendations(self, moderation_result) -> List[Dict]:
        """Generate actionable recommendations"""
        
        recommendations = []
        violations = moderation_result.violations
        
        # Critical recommendations
        critical_violations = [v for v in violations if v.severity == 'critical']
        if critical_violations:
            recommendations.append({
                'priority': 'critical',
                'title': 'Immediate Content Review Required',
                'description': f'Found {len(critical_violations)} critical violations that require immediate attention and content replacement.',
                'actions': [
                    'Review all flagged critical content',
                    'Replace inappropriate images with culturally appropriate alternatives',
                    'Consider document restructuring if needed'
                ]
            })
        
        # Cultural compliance recommendations
        cultural_violations = [
            v for v in violations 
            if v.category in ['adult_content', 'alcohol', 'gambling', 'revealing_clothing']
        ]
        
        if cultural_violations:
            recommendations.append({
                'priority': 'high',
                'title': 'Cultural Compliance Enhancement',
                'description': f'Document contains {len(cultural_violations)} instances that may not align with Islamic/Middle Eastern cultural values.',
                'actions': [
                    'Replace alcohol references with traditional beverages',
                    'Ensure modest dress code in all imagery', 
                    'Review content for cultural sensitivity',
                    'Consider local cultural guidelines'
                ]
            })
        
        # General improvements
        if len(violations) > 5:
            recommendations.append({
                'priority': 'medium',
                'title': 'Content Quality Improvement',
                'description': 'Multiple content issues detected that could benefit from comprehensive review.',
                'actions': [
                    'Implement content guidelines',
                    'Use AI-assisted content replacement',
                    'Establish review workflows',
                    'Regular compliance checking'
                ]
            })
        
        # Positive reinforcement
        if not violations:
            recommendations.append({
                'priority': 'low',
                'title': 'Content Approved',
                'description': 'Document meets cultural compliance standards.',
                'actions': [
                    'Document approved for use',
                    'Consider as template for future content',
                    'Regular periodic reviews recommended'
                ]
            })
        
        return recommendations
    
    def _base64_to_pil(self, base64_string: str) -> Optional[Image.Image]:
        """Convert base64 string to PIL Image"""
        try:
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]
            
            image_bytes = base64.b64decode(base64_string)
            return Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            logger.error(f"Failed to convert base64 to PIL: {e}")
            return None
    
    def _pil_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        try:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{img_base64}"
        except Exception as e:
            logger.error(f"Failed to convert PIL to base64: {e}")
            return ""
    
    def _result_to_dict(self, result) -> Dict:
        """Convert ModerationResult to dictionary for JSON serialization"""
        from dataclasses import asdict
        
        try:
            result_dict = asdict(result)
            
            # Handle any non-serializable objects
            for img_result in result_dict.get("image_results", []):
                # Truncate large base64 images for JSON export
                if "image_base64" in img_result and len(img_result["image_base64"]) > 2000:
                    img_result["image_base64"] = img_result["image_base64"][:2000] + "...[truncated]"
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Failed to convert result to dict: {e}")
            return {"error": str(e)}

# Create comprehensive report template if it doesn't exist
def create_comprehensive_template():
    """Create comprehensive report template"""
    
    template_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive Content Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        .header { background: #2c3e50; color: white; padding: 2rem; text-align: center; }
        .risk-critical { background: #e74c3c; color: white; padding: 1rem; }
        .risk-high { background: #e67e22; color: white; padding: 1rem; }
        .risk-medium { background: #f39c12; color: white; padding: 1rem; }
        .risk-low { background: #27ae60; color: white; padding: 1rem; }
        .section { margin: 2rem 0; padding: 1rem; border-left: 4px solid #3498db; }
        .violation { margin: 1rem 0; padding: 1rem; border-radius: 5px; }
        .violation-critical { background: #fee; border-left: 4px solid #e74c3c; }
        .violation-high { background: #fef9e7; border-left: 4px solid #e67e22; }
        .comparison-images { display: flex; gap: 2rem; margin: 2rem 0; }
        .comparison-images img { max-width: 300px; border-radius: 8px; }
        .highlight-high { background: #ffeb3b; font-weight: bold; padding: 2px; }
        .highlight-medium { background: #fff3cd; padding: 2px; }
        .highlight-low { background: #d1ecf1; padding: 2px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }
        .stat-card { background: #f8f9fa; padding: 1rem; text-align: center; border-radius: 8px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Content Moderation Analysis Report</h1>
        <h2>{{ result.file_name }}</h2>
        <div class="risk-{{ result.overall_risk_level }}">
            Risk Level: {{ result.overall_risk_level|upper }}
        </div>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <h3>{{ result.total_violations }}</h3>
                <p>Total Violations</p>
            </div>
            <div class="stat-card">
                <h3>{{ result.total_images }}</h3>
                <p>Images Analyzed</p>
            </div>
            <div class="stat-card">
                <h3>{{ (result.overall_confidence * 100)|round }}%</h3>
                <p>Confidence</p>
            </div>
            <div class="stat-card">
                <h3>{{ result.total_pages }}</h3>
                <p>Pages Processed</p>
            </div>
        </div>
    </div>

    {% if result.violations %}
    <div class="section">
        <h2>Detected Violations</h2>
        {% for violation in result.violations %}
        <div class="violation violation-{{ violation.severity }}">
            <h4>{{ violation.category|title|replace('_', ' ') }} ({{ violation.severity|upper }})</h4>
            <p><strong>Page:</strong> {{ violation.page_number }}</p>
            <p><strong>Description:</strong> {{ violation.description }}</p>
            <p><strong>Confidence:</strong> {{ (violation.confidence * 100)|round }}%</p>
        </div>
        {% endfor %}
    </div>
    {% endif %}

    {% if result.image_results %}
    <div class="section">
        <h2>Image Analysis with Replacements</h2>
        {% for image_result in result.image_results %}
        {% if image_result.replacement_image %}
        <h3>Page {{ image_result.page_number }} - Image {{ image_result.image_index }}</h3>
        <div class="comparison-images">
            <div>
                <h4>Original (Issue Detected)</h4>
                <img src="{{ image_result.image_base64 }}" alt="Original">
                <p>{{ image_result.caption }}</p>
            </div>
            <div>
                <h4>Culturally Appropriate Replacement</h4>
                <img src="{{ image_result.replacement_image.image_base64 }}" alt="Replacement">
                <p>Generated using {{ image_result.replacement_image.generation_method }}</p>
            </div>
        </div>
        {% endif %}
        {% endfor %}
    </div>
    {% endif %}

    <div class="section">
        <h2>Recommendations</h2>
        {% for recommendation in recommendations %}
        <div class="violation violation-{{ recommendation.priority }}">
            <h4>{{ recommendation.title }}</h4>
            <p>{{ recommendation.description }}</p>
            <ul>
                {% for action in recommendation.actions %}
                <li>{{ action }}</li>
                {% endfor %}
            </ul>
        </div>
        {% endfor %}
    </div>

    <div class="section">
        <h2>Processing Information</h2>
        <p><strong>Generated:</strong> {{ generation_timestamp|strftime }}</p>
        <p><strong>Processing Time:</strong> {{ result.processing_metadata.processing_time_seconds|round(2) }} seconds</p>
        <p><strong>File Size:</strong> {{ result.processing_metadata.pdf_file_size_mb }} MB</p>
        <p><strong>Models Used:</strong> {{ result.processing_metadata.models_used|join(', ') }}</p>
    </div>
</body>
</html>'''
    
    template_path = "templates/comprehensive_report.html"
    os.makedirs(os.path.dirname(template_path), exist_ok=True)
    
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(template_content)
    
    logger.info(f"Created comprehensive report template: {template_path}")

# Example usage
if __name__ == "__main__":
    # Create template if it doesn't exist
    create_comprehensive_template()
    
    print("Enhanced Report Generator ready!")
    print("Features:")
    print("- Comprehensive HTML/PDF reports")
    print("- Before/after image comparisons") 
    print("- Cultural compliance assessment")
    print("- Executive summaries and recommendations")
    print("- JSON data export")
    print("- Visual highlighting and formatting")