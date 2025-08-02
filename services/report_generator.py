# services/report_generator.py
import os
from datetime import datetime
from typing import Dict, List, Optional
import json
import base64
from io import BytesIO

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import HexColor, black, red, orange, green
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from utils.logger import get_logger
from utils.helpers import format_timestamp, format_file_size, truncate_text
from utils.exceptions import ContentModerationError

logger = get_logger(__name__)

class ReportGenerator:
    """Generates detailed reports from content moderation results"""
    
    def __init__(self):
        self.report_styles = self._create_styles()
        logger.info("Report Generator initialized")
    
    def _create_styles(self) -> Dict:
        """Create custom styles for reports"""
        if not REPORTLAB_AVAILABLE:
            return {}
        
        styles = getSampleStyleSheet()
        
        custom_styles = {
            'title': ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontSize=24,
                spaceAfter=30,
                textColor=HexColor('#2c3e50'),
                alignment=TA_CENTER
            ),
            'heading1': ParagraphStyle(
                'CustomHeading1',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=20,
                spaceBefore=20,
                textColor=HexColor('#34495e')
            ),
            'heading2': ParagraphStyle(
                'CustomHeading2',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=15,
                spaceBefore=15,
                textColor=HexColor('#34495e')
            ),
            'normal': ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=10,
                leading=14
            ),
            'risk_high': ParagraphStyle(
                'RiskHigh',
                parent=styles['Normal'],
                fontSize=11,
                textColor=HexColor('#e74c3c'),
                spaceAfter=8
            ),
            'risk_medium': ParagraphStyle(
                'RiskMedium',
                parent=styles['Normal'],
                fontSize=11,
                textColor=HexColor('#f39c12'),
                spaceAfter=8
            ),
            'risk_low': ParagraphStyle(
                'RiskLow',
                parent=styles['Normal'],
                fontSize=11,
                textColor=HexColor('#27ae60'),
                spaceAfter=8
            ),
            'summary_box': ParagraphStyle(
                'SummaryBox',
                parent=styles['Normal'],
                fontSize=12,
                spaceAfter=15,
                leftIndent=20,
                rightIndent=20,
                borderWidth=1,
                borderColor=HexColor('#bdc3c7')
            )
        }
        
        return custom_styles
    
    def generate_pdf_report(self, result_data: Dict, file_id: str) -> str:
        """Generate comprehensive PDF report"""
        if not REPORTLAB_AVAILABLE:
            logger.error("ReportLab not available. Cannot generate PDF reports.")
            # Generate HTML report as fallback
            return self.generate_html_report(result_data, file_id)
        
        try:
            # Ensure directory exists - use absolute path
            import os
            reports_dir = os.path.join(os.getcwd(), "static", "uploads")
            os.makedirs(reports_dir, exist_ok=True)
            
            output_path = os.path.join(reports_dir, f"report_{file_id}.pdf")
            
            # Create PDF document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Build content
            story = []
            
            # Title page
            story.extend(self._create_title_page(result_data))
            story.append(PageBreak())
            
            # Executive summary
            story.extend(self._create_executive_summary(result_data))
            story.append(PageBreak())
            
            # Detailed analysis
            story.extend(self._create_detailed_analysis(result_data))
            
            # Image analysis with detected images
            if result_data.get('image_results'):
                story.append(PageBreak())
                story.extend(self._create_image_analysis_section(result_data))
            
            # Violation details
            if result_data.get('violations'):
                story.append(PageBreak())
                story.extend(self._create_violation_details(result_data))
            
            # Recommendations
            story.append(PageBreak())
            story.extend(self._create_recommendations(result_data))
            
            # Appendix
            story.append(PageBreak())
            story.extend(self._create_appendix(result_data))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"PDF report generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            raise ContentModerationError(f"Report generation failed: {str(e)}")
    
    def _create_title_page(self, result_data: Dict) -> List:
        """Create professional title page elements"""
        elements = []
        
        # Professional Header with Logo space
        elements.append(Spacer(1, 0.5*inch))
        elements.append(Paragraph("DocShield Pro", self.report_styles['title']))
        elements.append(Paragraph("Professional Content Analysis Report", self.report_styles['heading1']))
        elements.append(Spacer(1, 0.5*inch))
        
        # Executive Summary Box
        elements.append(Paragraph("EXECUTIVE SUMMARY", self.report_styles['heading1']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Key metrics in a professional layout
        doc_info = [
            ["Document Name:", result_data.get('file_name', 'Unknown Document')],
            ["Analysis Completed:", format_timestamp(result_data.get('processing_timestamp'))],
            ["Report Reference:", result_data.get('document_id', 'Unknown')[:12]],
            ["Risk Assessment:", result_data.get('overall_risk_level', 'Unknown').upper()],
            ["Analysis Confidence:", f"{(result_data.get('overall_confidence', 0)*100):.0f}%"],
            ["Issues Identified:", str(result_data.get('total_violations', 0))],
            ["Images Processed:", str(result_data.get('total_images', 0))],
            ["Pages Analyzed:", str(result_data.get('total_pages', 0))]
        ]
        
        info_table = Table(doc_info, colWidths=[2*inch, 3*inch])
        info_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [None, HexColor('#f8f9fa')]),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#dee2e6'))
        ]))
        
        elements.append(info_table)
        elements.append(Spacer(1, 1*inch))
        
        # Risk level indicator
        risk_level = result_data.get('overall_risk_level', 'unknown').lower()
        risk_color = {
            'high': HexColor('#e74c3c'),
            'medium': HexColor('#f39c12'),
            'low': HexColor('#27ae60')
        }.get(risk_level, black)
        
        risk_style = ParagraphStyle(
            'RiskIndicator',
            parent=self.report_styles['normal'],
            fontSize=18,
            textColor=risk_color,
            alignment=TA_CENTER
        )
        
        elements.append(Paragraph(f"Risk Level: {risk_level.upper()}", risk_style))
        
        return elements
    
    def _create_executive_summary(self, result_data: Dict) -> List:
        """Create executive summary section"""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.report_styles['heading1']))
        
        # Summary statistics
        summary_stats = result_data.get('summary_stats', {})
        
        summary_text = f"""
        This document presents the results of automated content moderation analysis performed on 
        "{result_data.get('file_name', 'the submitted document')}". The analysis examined 
        {result_data.get('total_images', 0)} images and {result_data.get('total_pages', 0)} pages 
        of text content using advanced AI models including Florence-2 for image captioning, 
        CLIP for visual analysis, and transformer-based NLP models for text analysis.
        """
        
        elements.append(Paragraph(summary_text, self.report_styles['normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Key findings
        elements.append(Paragraph("Key Findings:", self.report_styles['heading2']))
        
        violations = result_data.get('violations', [])
        violation_stats = summary_stats.get('violation_stats', {})
        
        findings = [
            f"• Total violations detected: {len(violations)}",
            f"• High-risk violations: {violation_stats.get('violations_by_severity', {}).get('high', 0)}",
            f"• Medium-risk violations: {violation_stats.get('violations_by_severity', {}).get('medium', 0)}",
            f"• Low-risk violations: {violation_stats.get('violations_by_severity', {}).get('low', 0)}",
            f"• Overall confidence score: {result_data.get('overall_confidence', 0):.2f}"
        ]
        
        for finding in findings:
            elements.append(Paragraph(finding, self.report_styles['normal']))
        
        return elements
    
    def _create_detailed_analysis(self, result_data: Dict) -> List:
        """Create detailed analysis section"""
        elements = []
        
        elements.append(Paragraph("Detailed Analysis", self.report_styles['heading1']))
        
        # Image analysis
        if result_data.get('image_results'):
            elements.append(Paragraph("Image Analysis", self.report_styles['heading2']))
            
            image_stats = result_data.get('summary_stats', {}).get('image_stats', {})
            vision_stats = result_data.get('summary_stats', {}).get('vision_stats', {})
            
            image_analysis_text = f"""
            Analyzed {image_stats.get('total_images', 0)} images with an average NSFW score of 
            {vision_stats.get('avg_nsfw_score', 0):.3f}. {vision_stats.get('high_risk_images', 0)} 
            images were classified as high-risk content.
            """
            
            elements.append(Paragraph(image_analysis_text, self.report_styles['normal']))
            
            # Top detected categories
            top_categories = vision_stats.get('detected_categories', [])
            if top_categories:
                elements.append(Paragraph("Most Detected Visual Categories:", self.report_styles['normal']))
                for category, score in top_categories[:5]:
                    elements.append(Paragraph(f"• {category.title()}: {score:.3f}", self.report_styles['normal']))
        
        # Text analysis
        if result_data.get('text_results'):
            elements.append(Spacer(1, 0.2*inch))
            elements.append(Paragraph("Text Analysis", self.report_styles['heading2']))
            
            text_stats = result_data.get('summary_stats', {}).get('text_stats', {})
            
            text_analysis_text = f"""
            Analyzed {text_stats.get('total_words', 0)} words across {text_stats.get('total_pages_with_text', 0)} 
            pages. Detected {text_stats.get('total_risk_keywords', 0)} risk keywords with an average 
            text quality score of {text_stats.get('avg_text_quality', 0):.2f}.
            """
            
            elements.append(Paragraph(text_analysis_text, self.report_styles['normal']))
            
            # Languages detected
            languages = text_stats.get('languages_detected', [])
            if languages:
                elements.append(Paragraph(f"Languages detected: {', '.join(languages)}", self.report_styles['normal']))
        
        return elements
    
    def _create_image_analysis_section(self, result_data: Dict) -> List:
        """Create comprehensive image analysis section with detected images"""
        elements = []
        
        elements.append(Paragraph("IMAGE CONTENT ANALYSIS", self.report_styles['heading1']))
        
        image_results = result_data.get('image_results', [])
        if not image_results:
            elements.append(Paragraph("No images were detected in the document.", self.report_styles['normal']))
            return elements
        
        # Statistics
        total_images = len(image_results)
        flagged_images = len([img for img in image_results if img.get('violations')])
        clean_images = total_images - flagged_images
        
        stats_text = f"""
        📊 Image Analysis Summary:
        • Total images detected: {total_images}
        • Flagged for violations: {flagged_images}
        • Approved content: {clean_images}
        • Overall compliance rate: {(clean_images/total_images)*100:.1f}%
        """
        
        elements.append(Paragraph(stats_text, self.report_styles['summary_box']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Separate flagged and clean images
        flagged_imgs = [img for img in image_results if img.get('violations')]
        clean_imgs = [img for img in image_results if not img.get('violations')]
        
        # Unacceptable Images Section
        if flagged_imgs:
            elements.append(Paragraph("🚫 Unacceptable Images - Policy Violations", self.report_styles['heading2']))
            elements.append(Paragraph("⚠️ WARNING: The following images violate Middle Eastern cultural standards and Islamic principles.", self.report_styles['risk_high']))
            
            for i, img_result in enumerate(flagged_imgs[:5]):  # Limit to 5 flagged images
                elements.append(Paragraph(f"Unacceptable Image #{i+1} - Policy Violation (Page {img_result.get('page_number', 'N/A')})", 
                                        self.report_styles['risk_high']))
                
                # Add image if base64 data exists
                if img_result.get('image_base64'):
                    try:
                        # Decode base64 image
                        img_data = img_result['image_base64']
                        if img_data.startswith('data:image'):
                            img_data = img_data.split(',')[1]
                        
                        # Create ReportLab Image from base64
                        img_bytes = base64.b64decode(img_data)
                        img_buffer = BytesIO(img_bytes)
                        
                        # Add image to report (with size constraints)
                        img = Image(img_buffer, width=2*inch, height=1.5*inch)
                        elements.append(img)
                        
                    except Exception as e:
                        logger.warning(f"Failed to add image to report: {e}")
                        elements.append(Paragraph("⚠️ Image data could not be displayed", self.report_styles['normal']))
                
                # Add AI-generated description
                caption = img_result.get('caption', 'No description available')
                if caption and caption != 'No description available':
                    description_text = f"AI Analysis: {caption}"
                    elements.append(Paragraph(description_text, self.report_styles['normal']))
                
                # Add violation details
                violations = img_result.get('violations', [])
                if violations:
                    violation_text = f"Policy Violation: {violations[0].get('description', 'Inappropriate content')}"
                    elements.append(Paragraph(violation_text, self.report_styles['risk_high']))
                
                elements.append(Spacer(1, 0.1*inch))
        
        # Culturally Compliant Images Section
        if clean_imgs:
            elements.append(Paragraph("✅ Culturally Compliant Images", self.report_styles['heading2']))
            elements.append(Paragraph(f"The following {len(clean_imgs)} images were analyzed and found to be compliant with cultural standards:", 
                                    self.report_styles['normal']))
            
            # Show actual approved images (limit to 8 for space)
            for i, img_result in enumerate(clean_imgs[:8]):
                elements.append(Paragraph(f"Approved Image #{i+1} (Page {img_result.get('page_number', 'N/A')})", 
                                        self.report_styles['normal']))
                
                # Add image if base64 data exists
                if img_result.get('image_base64'):
                    try:
                        # Decode base64 image
                        img_data = img_result['image_base64']
                        if img_data.startswith('data:image'):
                            img_data = img_data.split(',')[1]
                        
                        # Create ReportLab Image from base64
                        img_bytes = base64.b64decode(img_data)
                        img_buffer = BytesIO(img_bytes)
                        
                        # Add image to report (slightly smaller for approved images)
                        img = Image(img_buffer, width=1.5*inch, height=1.2*inch)
                        elements.append(img)
                        
                    except Exception as e:
                        logger.warning(f"Failed to add approved image to report: {e}")
                        elements.append(Paragraph("⚠️ Image data could not be displayed", self.report_styles['normal']))
                
                # Add AI-generated description
                caption = img_result.get('caption', 'No description available')
                if caption and caption != 'No description available':
                    description_text = f"AI Analysis: {caption}"
                    elements.append(Paragraph(description_text, self.report_styles['normal']))
                else:
                    elements.append(Paragraph("AI Analysis: Content analysis complete - no description generated", self.report_styles['normal']))
                
                elements.append(Spacer(1, 0.1*inch))
                
            # Summary table for remaining approved images if there are more than 8
            if len(clean_imgs) > 8:
                elements.append(Paragraph(f"Additional {len(clean_imgs) - 8} approved images:", self.report_styles['normal']))
                
                remaining_table_data = [['Page', 'Status', 'Caption Summary']]
                
                for img_result in clean_imgs[8:]:  # Show remaining images in table
                    caption = img_result.get('caption', 'No caption available')
                    if len(caption) > 50:
                        caption = caption[:50] + "..."
                    
                    remaining_table_data.append([
                        str(img_result.get('page_number', 'N/A')),
                        '✅ Compliant',
                        caption
                    ])
                
                if len(remaining_table_data) > 1:  # Has data beyond header
                    remaining_table = Table(remaining_table_data, colWidths=[1*inch, 1.5*inch, 3.5*inch])
                    remaining_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#27ae60')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#f0f8f0'), None]),
                        ('GRID', (0, 0), (-1, -1), 1, HexColor('#27ae60'))
                    ]))
                    
                    elements.append(remaining_table)
        
        # Cultural Compliance Analysis
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph("🏛️ Cultural Compliance Analysis", self.report_styles['heading2']))
        
        compliance_text = f"""
        Based on Middle Eastern cultural values and Islamic principles:
        • {clean_images} images meet regional cultural standards
        • {flagged_images} images require review or replacement
        • All images have been analyzed for modesty and appropriateness
        • Content recommendations provided for non-compliant imagery
        """
        
        elements.append(Paragraph(compliance_text, self.report_styles['normal']))
        
        return elements
    
    def _create_violation_details(self, result_data: Dict) -> List:
        """Create violation details section"""
        elements = []
        
        elements.append(Paragraph("Violation Details", self.report_styles['heading1']))
        
        violations = result_data.get('violations', [])
        
        if not violations:
            elements.append(Paragraph("No violations detected.", self.report_styles['normal']))
            return elements
        
        # Group violations by severity
        violation_groups = {'high': [], 'medium': [], 'low': []}
        for violation in violations:
            severity = violation.get('severity', 'low')
            violation_groups[severity].append(violation)
        
        # Create tables for each severity level
        for severity in ['high', 'medium', 'low']:
            if violation_groups[severity]:
                elements.append(Paragraph(f"{severity.title()} Risk Violations", self.report_styles['heading2']))
                
                # Create table data
                table_data = [['Page', 'Type', 'Category', 'Description', 'Confidence']]
                
                for violation in violation_groups[severity][:10]:  # Limit to 10 per severity
                    table_data.append([
                        str(violation.get('page_number', 'N/A')),
                        violation.get('violation_type', 'N/A'),
                        violation.get('category', 'N/A'),
                        truncate_text(violation.get('description', 'N/A'), 50),
                        f"{violation.get('confidence', 0):.2f}"
                    ])
                
                # Create table
                violation_table = Table(table_data, colWidths=[0.8*inch, 0.8*inch, 1.2*inch, 2.5*inch, 0.8*inch])
                violation_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#34495e')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#f8f9fa'), None]),
                    ('GRID', (0, 0), (-1, -1), 1, HexColor('#dee2e6'))
                ]))
                
                elements.append(violation_table)
                elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _create_recommendations(self, result_data: Dict) -> List:
        """Create recommendations section"""
        elements = []
        
        elements.append(Paragraph("Recommendations", self.report_styles['heading1']))
        
        violations = result_data.get('violations', [])
        risk_level = result_data.get('overall_risk_level', 'low')
        
        # Generate Middle East cultural compliance recommendations
        recommendations = []
        
        if risk_level == 'high':
            recommendations.extend([
                "⚠️ URGENT: This document contains content that may violate Middle Eastern cultural standards and local customs.",
                "🏛️ Immediate review required to ensure compliance with regional cultural values and traditions.",
                "📋 Consider content removal or modification to align with local cultural guidelines."
            ])
        
        if len(violations) > 10:
            recommendations.append("The document contains numerous content violations. Consider comprehensive content review.")
        
        # Middle East cultural specific category recommendations
        violation_categories = {}
        for violation in violations:
            category = violation.get('category', 'unknown')
            violation_categories[category] = violation_categories.get(category, 0) + 1
        
        for category, count in violation_categories.items():
            if count >= 3:
                if category.lower() in ['nsfw', 'inappropriate', 'explicit']:
                    recommendations.append(f"🚫 {count} instances of inappropriate content detected that conflicts with regional cultural values. Immediate content review required.")
                elif category.lower() in ['suggestive', 'revealing']:
                    recommendations.append(f"⚠️ {count} instances of immodest content found. Please ensure compliance with Middle Eastern cultural standards.")
                else:
                    recommendations.append(f"📊 Multiple instances of {category} content detected ({count} violations). Review against local cultural guidelines.")
        
        # Image-specific recommendations
        image_violations = [v for v in violations if v.get('violation_type') == 'image']
        if len(image_violations) > 5:
            recommendations.append("Consider implementing stricter image content policies.")
        
        # Text-specific recommendations
        text_violations = [v for v in violations if v.get('violation_type') == 'text']
        if len(text_violations) > 5:
            recommendations.append("Review text content for policy compliance and consider content editing.")
        
        # Default recommendations
        if not recommendations:
            if risk_level == 'medium':
                recommendations.append("Document contains moderate risk content. Review flagged sections.")
            elif risk_level == 'low':
                recommendations.append("Document appears to be compliant with content policies.")
            else:
                recommendations.append("Review document content against organizational policies.")
        
        # Add Middle East cultural compliance general recommendations
        recommendations.extend([
            "🏛️ Implement regular content auditing based on Middle Eastern cultural values and traditions.",
            "📚 Provide training on regional cultural guidelines and local sensitivities to content creators.",
            "🔍 Consider implementing automated content filtering that respects local customs and traditions.",
            "🤝 Establish content review committees with regional cultural experts and community leaders.",
            "📖 Ensure all content aligns with local cultural standards and community values."
        ])
        
        for i, recommendation in enumerate(recommendations, 1):
            elements.append(Paragraph(f"{i}. {recommendation}", self.report_styles['normal']))
        
        return elements
    
    def _create_appendix(self, result_data: Dict) -> List:
        """Create appendix section"""
        elements = []
        
        elements.append(Paragraph("Appendix", self.report_styles['heading1']))
        
        # Technical details
        elements.append(Paragraph("Technical Details", self.report_styles['heading2']))
        
        processing_metadata = result_data.get('processing_metadata', {})
        
        tech_details = [
            f"• Processing time: {processing_metadata.get('processing_time_seconds', 0):.2f} seconds",
            f"• Models used: {', '.join(processing_metadata.get('models_used', []))}",
            f"• Confidence threshold: {processing_metadata.get('confidence_threshold', 0.7)}",
            f"• Processing device: {processing_metadata.get('processing_device', 'Unknown')}"
        ]
        
        for detail in tech_details:
            elements.append(Paragraph(detail, self.report_styles['normal']))
        
        # Glossary
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph("Glossary", self.report_styles['heading2']))
        
        glossary = [
            "• NSFW: Not Safe For Work content",
            "• Confidence Score: AI model's certainty in its prediction (0.0-1.0)",
            "• Florence-2: Microsoft's vision-language model for image captioning",
            "• CLIP: OpenAI's model for understanding images and text",
            "• Risk Level: Overall assessment of content appropriateness"
        ]
        
        for term in glossary:
            elements.append(Paragraph(term, self.report_styles['normal']))
        
        return elements
    
    def generate_html_report(self, result_data: Dict, file_id: str) -> str:
        """Generate HTML report as fallback"""
        try:
            reports_dir = os.path.join(os.getcwd(), "static", "uploads")
            os.makedirs(reports_dir, exist_ok=True)
            output_path = os.path.join(reports_dir, f"report_{file_id}.html")
            
            html_content = self._create_html_content(result_data)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML report generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            raise ContentModerationError(f"HTML report generation failed: {str(e)}")
    
    def _create_html_content(self, result_data: Dict) -> str:
        """Create HTML content for report"""
        violations = result_data.get('violations', [])
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Content Moderation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .risk-high {{ color: #e74c3c; }}
                .risk-medium {{ color: #f39c12; }}
                .risk-low {{ color: #27ae60; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Content Moderation Report</h1>
                <h2 class="risk-{result_data.get('overall_risk_level', 'low')}">
                    Risk Level: {result_data.get('overall_risk_level', 'unknown').upper()}
                </h2>
            </div>
            
            <h3>Document Information</h3>
            <p><strong>File:</strong> {result_data.get('file_name', 'Unknown')}</p>
            <p><strong>Analysis Date:</strong> {format_timestamp(result_data.get('processing_timestamp'))}</p>
            <p><strong>Total Violations:</strong> {len(violations)}</p>
            <p><strong>Confidence:</strong> {result_data.get('overall_confidence', 0):.2f}</p>
            
            <h3>Violations</h3>
        """
        
        if violations:
            html += """
            <table>
                <tr>
                    <th>Page</th>
                    <th>Type</th>
                    <th>Category</th>
                    <th>Severity</th>
                    <th>Description</th>
                    <th>Confidence</th>
                </tr>
            """
            
            for violation in violations:
                severity_class = f"risk-{violation.get('severity', 'low')}"
                html += f"""
                <tr>
                    <td>{violation.get('page_number', 'N/A')}</td>
                    <td>{violation.get('violation_type', 'N/A')}</td>
                    <td>{violation.get('category', 'N/A')}</td>
                    <td class="{severity_class}">{violation.get('severity', 'N/A').upper()}</td>
                    <td>{violation.get('description', 'N/A')}</td>
                    <td>{violation.get('confidence', 0):.2f}</td>
                </tr>
                """
            
            html += "</table>"
        else:
            html += "<p>No violations detected.</p>"
        
        html += """
            </body>
        </html>
        """
        
        return html
    
    def generate_json_summary(self, result_data: Dict, file_id: str) -> str:
        """Generate JSON summary report"""
        try:
            reports_dir = os.path.join(os.getcwd(), "static", "uploads")
            os.makedirs(reports_dir, exist_ok=True)
            output_path = os.path.join(reports_dir, f"summary_{file_id}.json")
            
            summary = {
                'report_id': file_id,
                'generated_at': datetime.now().isoformat(),
                'document_info': {
                    'filename': result_data.get('file_name'),
                    'analysis_date': result_data.get('processing_timestamp'),
                    'total_pages': result_data.get('total_pages'),
                    'total_images': result_data.get('total_images')
                },
                'risk_assessment': {
                    'overall_risk_level': result_data.get('overall_risk_level'),
                    'confidence': result_data.get('overall_confidence'),
                    'total_violations': result_data.get('total_violations')
                },
                'violation_summary': self._create_violation_summary(result_data.get('violations', [])),
                'processing_info': result_data.get('processing_metadata', {})
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"JSON summary generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to generate JSON summary: {e}")
            raise ContentModerationError(f"JSON summary generation failed: {str(e)}")
    
    def _create_violation_summary(self, violations: List[Dict]) -> Dict:
        """Create violation summary statistics"""
        if not violations:
            return {'total': 0}
        
        summary = {
            'total': len(violations),
            'by_severity': {'high': 0, 'medium': 0, 'low': 0},
            'by_type': {'image': 0, 'text': 0},
            'by_category': {},
            'by_page': {}
        }
        
        for violation in violations:
            # By severity
            severity = violation.get('severity', 'low')
            summary['by_severity'][severity] += 1
            
            # By type
            v_type = violation.get('violation_type', 'unknown')
            if v_type in summary['by_type']:
                summary['by_type'][v_type] += 1
            
            # By category
            category = violation.get('category', 'unknown')
            summary['by_category'][category] = summary['by_category'].get(category, 0) + 1
            
            # By page
            page = str(violation.get('page_number', 'unknown'))
            summary['by_page'][page] = summary['by_page'].get(page, 0) + 1
        
        return summary