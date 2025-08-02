# core/content_moderator.py
import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import logging

from core.pdf_processor import PDFProcessor, ProcessedImage
from core.vision_analyzer import VisionAnalyzer, VisionAnalysisResult
from core.nlp_analyzer import NLPAnalyzer, TextAnalysisResult
from utils.logger import get_logger
# Moved import inside methods to avoid circular import

logger = get_logger(__name__)

@dataclass
class ViolationReport:
    """Individual content violation report"""
    violation_id: str
    violation_type: str  # 'image', 'text', 'combined'
    page_number: int
    severity: str  # 'high', 'medium', 'low'
    confidence: float
    category: str
    description: str
    evidence: Dict  # Supporting evidence (image, text, etc.)
    risk_factors: List[str]
    timestamp: str

@dataclass
class ModerationResult:
    """Complete moderation result for a PDF"""
    document_id: str
    file_name: str
    processing_timestamp: str
    total_pages: int
    total_images: int
    total_violations: int
    overall_risk_level: str
    overall_confidence: float
    
    # Detailed results
    image_results: List[ProcessedImage]
    text_results: List[TextAnalysisResult]
    vision_results: List[VisionAnalysisResult]
    violations: List[ViolationReport]
    
    # Summary statistics
    summary_stats: Dict
    processing_metadata: Dict

class ContentModerator:
    """Main content moderation orchestrator"""
    
    def __init__(self):
        # Initialize components with error handling
        try:
            self.pdf_processor = PDFProcessor()
            logger.info("✅ PDF processor initialized")
        except Exception as e:
            logger.error(f"❌ PDF processor failed: {e}")
            raise
            
        try:
            self.vision_analyzer = VisionAnalyzer()
            logger.info("✅ Vision analyzer initialized (with available models)")
        except Exception as e:
            logger.error(f"❌ Vision analyzer failed: {e}")
            self.vision_analyzer = None
            
        try:
            self.nlp_analyzer = NLPAnalyzer()
            logger.info("✅ NLP analyzer initialized")
        except Exception as e:
            logger.error(f"❌ NLP analyzer failed: {e}")
            self.nlp_analyzer = None
        
        # Import inside __init__ to avoid circular import
        try:
            from app.config import app_config
            self.confidence_threshold = app_config.confidence_threshold
        except:
            self.confidence_threshold = 0.7
            
        logger.info("Content Moderator initialized with working components")
    
    def moderate_pdf(self, pdf_path: str, file_name: str = None) -> ModerationResult:
        """Complete PDF content moderation pipeline"""
        start_time = datetime.now()
        document_id = str(uuid.uuid4())
        
        if file_name is None:
            file_name = os.path.basename(pdf_path)
        
        logger.info(f"Starting content moderation for: {file_name}")
        
        try:
            # Step 1: Process PDF and extract images
            logger.info("Step 1: Processing PDF and extracting images...")
            image_results = self.pdf_processor.process_pdf_complete(pdf_path)
            
            # Step 2: Analyze images with computer vision
            logger.info("Step 2: Analyzing images with computer vision...")
            vision_results = []
            
            if self.vision_analyzer:
                for img_result in image_results:
                    try:
                        # Convert base64 back to PIL Image for analysis
                        from PIL import Image
                        import io
                        import base64
                        
                        # Extract image data from base64
                        image_data = img_result.image_base64.split(',')[1]
                        image_bytes = base64.b64decode(image_data)
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        
                        vision_result = self.vision_analyzer.analyze_image(pil_image, img_result.caption)
                        vision_results.append(vision_result)
                    except Exception as e:
                        logger.error(f"Vision analysis failed for image: {e}")
                        # Create a basic result
                        from core.vision_analyzer import VisionAnalysisResult
                        vision_result = VisionAnalysisResult(
                            nsfw_score=0.0,
                            clip_similarities={},
                            scene_classification={},
                            detected_objects=[],
                            risk_level="low",
                            confidence=0.5,
                            analysis_details={"error": str(e)}
                        )
                        vision_results.append(vision_result)
            else:
                logger.warning("Vision analyzer not available - skipping image analysis")
                # Create basic results for each image
                from core.vision_analyzer import VisionAnalysisResult
                for img_result in image_results:
                    vision_result = VisionAnalysisResult(
                        nsfw_score=0.0,
                        clip_similarities={},
                        scene_classification={"general": 0.5},
                        detected_objects=[],
                        risk_level="low",
                        confidence=0.5,
                        analysis_details={"status": "vision_analyzer_unavailable"}
                    )
                    vision_results.append(vision_result)
            
            # Step 3: Analyze text content
            logger.info("Step 3: Analyzing text content...")
            if self.nlp_analyzer:
                try:
                    text_results = self.nlp_analyzer.analyze_pdf_text(pdf_path)
                except Exception as e:
                    logger.error(f"NLP analysis failed: {e}")
                    text_results = []
            else:
                logger.warning("NLP analyzer not available - skipping text analysis")
                text_results = []
            
            # Step 4: Generate violation reports and attach to results
            logger.info("Step 4: Generating violation reports...")
            violations = self._generate_violation_reports(
                image_results, vision_results, text_results
            )
            
            # Attach violations to individual results for template display
            self._attach_violations_to_results(image_results, text_results, violations)
            
            # Step 5: Calculate overall risk assessment
            logger.info("Step 5: Calculating overall risk assessment...")
            overall_risk, overall_confidence = self._calculate_overall_risk(
                vision_results, text_results, violations
            )
            
            # Step 6: Generate summary statistics
            summary_stats = self._generate_summary_stats(
                image_results, vision_results, text_results, violations
            )
            
            # Step 7: Create processing metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            processing_metadata = {
                "processing_time_seconds": processing_time,
                "pdf_file_size_mb": round(os.path.getsize(pdf_path) / (1024 * 1024), 2),
                "models_used": ["BLIP", "CLIP", "NSFW-Detector", "Sentence-Transformers"],
                "confidence_threshold": self.confidence_threshold,
                "processing_device": self.pdf_processor.vision_model.device
            }
            
            # Create final result
            result = ModerationResult(
                document_id=document_id,
                file_name=file_name,
                processing_timestamp=start_time.isoformat(),
                total_pages=len(text_results) if text_results else 0,
                total_images=len(image_results),
                total_violations=len(violations),
                overall_risk_level=overall_risk,
                overall_confidence=overall_confidence,
                image_results=image_results,
                text_results=text_results,
                vision_results=vision_results,
                violations=violations,
                summary_stats=summary_stats,
                processing_metadata=processing_metadata
            )
            
            logger.info(f"Content moderation completed. Found {len(violations)} violations with {overall_risk} risk level")
            return result
            
        except Exception as e:
            logger.error(f"Content moderation failed: {e}")
            raise RuntimeError(f"Moderation pipeline failed: {e}")
    
    def _generate_violation_reports(self, image_results: List[ProcessedImage], 
                                  vision_results: List[VisionAnalysisResult],
                                  text_results: List[TextAnalysisResult]) -> List[ViolationReport]:
        """Generate detailed violation reports"""
        violations = []
        
        # Image-based violations
        for img_result, vision_result in zip(image_results, vision_results):
            img_violations = self._analyze_image_violations(img_result, vision_result)
            violations.extend(img_violations)
        
        # Text-based violations
        for text_result in text_results:
            text_violations = self._analyze_text_violations(text_result)
            violations.extend(text_violations)
        
        # Sort violations by severity and confidence
        violations.sort(key=lambda x: (
            {"high": 3, "medium": 2, "low": 1}[x.severity],
            x.confidence
        ), reverse=True)
        
        return violations
    
    def _attach_violations_to_results(self, image_results: List[ProcessedImage],
                                     text_results: List[TextAnalysisResult],
                                     violations: List[ViolationReport]):
        """Attach violations to individual image and text results for template display"""
        
        # Create lookup dictionaries for violations by page and type
        image_violations_by_page = {}
        text_violations_by_page = {}
        
        for violation in violations:
            page_num = violation.page_number
            if violation.violation_type == "image":
                if page_num not in image_violations_by_page:
                    image_violations_by_page[page_num] = []
                image_violations_by_page[page_num].append(violation)
            elif violation.violation_type == "text":
                if page_num not in text_violations_by_page:
                    text_violations_by_page[page_num] = []
                text_violations_by_page[page_num].append(violation)
        
        # Attach violations to image results
        for img_result in image_results:
            page_violations = image_violations_by_page.get(img_result.page_number, [])
            # Add violations attribute to the image result
            img_result.violations = page_violations
        
        # Attach violations to text results  
        for text_result in text_results:
            page_num = text_result.analysis_details.get("page_number", 1)
            page_violations = text_violations_by_page.get(page_num, [])
            # Add violations attribute to the text result
            text_result.violations = page_violations
    
    def _analyze_image_violations(self, img_result: ProcessedImage, 
                                vision_result: VisionAnalysisResult) -> List[ViolationReport]:
        """Analyze violations in a single image"""
        violations = []
        
        # NSFW violations
        if vision_result.nsfw_score > 0.5:
            severity = "high" if vision_result.nsfw_score > 0.7 else "medium"
            
            violation = ViolationReport(
                violation_id=str(uuid.uuid4()),
                violation_type="image",
                page_number=img_result.page_number,
                severity=severity,
                confidence=vision_result.nsfw_score,
                category="adult_content",
                description=f"NSFW content detected with confidence {vision_result.nsfw_score:.2f}",
                evidence={
                    "image_caption": img_result.caption,
                    "nsfw_score": vision_result.nsfw_score,
                    "image_size": img_result.size,
                    "image_base64": img_result.image_base64[:100] + "..."  # Truncated for storage
                },
                risk_factors=["nsfw_detection"],
                timestamp=datetime.now().isoformat()
            )
            violations.append(violation)
        
        # Enhanced cultural content category violations for Middle East
        risk_categories = {
            "adult content": "high",
            "revealing clothing": "high",  # Elevated for cultural compliance
            "inappropriate clothing": "medium",
            "alcohol": "high",  # Elevated for cultural compliance
            "gambling": "high",  # Elevated for cultural compliance
            "violence": "high",
            "weapons": "high",
            "dating scenes": "medium",  # Added for cultural sensitivity
            "intimate situations": "high",  # Added for cultural sensitivity
            "western holidays": "low",  # Added for cultural context
            "non-modest imagery": "medium",  # Added for cultural compliance
            "mixed gender socializing": "low"  # Added for cultural context
        }
        
        for category, severity in risk_categories.items():
            if category in vision_result.clip_similarities:
                confidence = vision_result.clip_similarities[category]
                
                # Define thresholds based on severity
                threshold = 0.3 if severity == "high" else 0.4 if severity == "medium" else 0.5
                
                if confidence > threshold:
                    violation = ViolationReport(
                        violation_id=str(uuid.uuid4()),
                        violation_type="image",
                        page_number=img_result.page_number,
                        severity=severity,
                        confidence=confidence,
                        category=category.replace(" ", "_"),
                        description=f"Detected {category} content in image with confidence {confidence:.2f}",
                        evidence={
                            "image_caption": img_result.caption,
                            "clip_similarity": confidence,
                            "detected_objects": vision_result.detected_objects,
                            "analysis_details": vision_result.analysis_details
                        },
                        risk_factors=[f"clip_detection_{category.replace(' ', '_')}"],
                        timestamp=datetime.now().isoformat()
                    )
                    violations.append(violation)
        
        # Caption-based violations
        caption_violations = self._analyze_caption_violations(
            img_result.caption, img_result.page_number, vision_result
        )
        violations.extend(caption_violations)
        
        return violations
    
    def _analyze_caption_violations(self, caption: str, page_number: int, 
                                  vision_result: VisionAnalysisResult) -> List[ViolationReport]:
        """Analyze violations in image captions"""
        violations = []
        
        if not caption:
            return violations
        
        caption_lower = caption.lower()
        
        # Check against content rules - avoid circular import
        try:
            from app.config import CONTENT_RULES
        except ImportError:
            # Fallback content rules if import fails
            CONTENT_RULES = {
                "high_risk": {
                    "adult_content": {
                        "keywords": ["nude", "naked", "explicit", "sexual"],
                        "confidence_threshold": 0.7,
                        "severity": "high"
                    }
                }
            }
        for categories in CONTENT_RULES.values():
            for category, rules in categories.items():
                for keyword in rules["keywords"]:
                    if keyword.lower() in caption_lower:
                        violation = ViolationReport(
                            violation_id=str(uuid.uuid4()),
                            violation_type="image",
                            page_number=page_number,
                            severity=rules["severity"],
                            confidence=rules["confidence_threshold"],
                            category=category,
                            description=f"Found prohibited keyword '{keyword}' in image caption",
                            evidence={
                                "caption": caption,
                                "keyword": keyword,
                                "category": category,
                                "vision_analysis": vision_result.analysis_details
                            },
                            risk_factors=[f"caption_keyword_{keyword}"],
                            timestamp=datetime.now().isoformat()
                        )
                        violations.append(violation)
        
        return violations
    
    def _analyze_text_violations(self, text_result: TextAnalysisResult) -> List[ViolationReport]:
        """Analyze violations in text content"""
        violations = []
        
        page_number = text_result.analysis_details.get("page_number", 1)
        
        # Risk keyword violations with context extraction
        full_text = text_result.analysis_details.get("text_content", "")
        
        for keyword_info in text_result.risk_keywords:
            # Extract context around the keyword
            keyword = keyword_info["keyword"]
            context = self._extract_keyword_context(full_text, keyword)
            
            violation = ViolationReport(
                violation_id=str(uuid.uuid4()),
                violation_type="text",
                page_number=page_number,
                severity=keyword_info["severity"],
                confidence=keyword_info["confidence"],
                category=keyword_info["category"],
                description=f"Found risk keyword '{keyword}' in text (count: {keyword_info['count']})",
                evidence={
                    "keyword": keyword,
                    "count": keyword_info["count"],
                    "context": context,
                    "text_analysis": text_result.analysis_details,
                    "full_context": f"...{context}..." if context else "Context not available"
                },
                risk_factors=[f"text_keyword_{keyword}"],
                timestamp=datetime.now().isoformat()
            )
            violations.append(violation)
        
        # High-confidence semantic violations
        semantic_threshold = 0.6
        risk_semantic_categories = {
            "adult_content": "high",
            "violence": "high", 
            "alcohol_drugs": "medium",
            "gambling": "medium",
            "dating_romance": "low"
        }
        
        for category, severity in risk_semantic_categories.items():
            if (category in text_result.semantic_similarities and 
                text_result.semantic_similarities[category] > semantic_threshold):
                
                confidence = text_result.semantic_similarities[category]
                
                violation = ViolationReport(
                    violation_id=str(uuid.uuid4()),
                    violation_type="text",
                    page_number=page_number,
                    severity=severity,
                    confidence=confidence,
                    category=category,
                    description=f"Semantic analysis detected {category} content with confidence {confidence:.2f}",
                    evidence={
                        "semantic_score": confidence,
                        "text_stats": {
                            "word_count": text_result.total_words,
                            "language": text_result.language,
                            "sentiment": text_result.sentiment_score
                        },
                        "entities": text_result.extracted_entities[:5]  # First 5 entities
                    },
                    risk_factors=[f"semantic_{category}"],
                    timestamp=datetime.now().isoformat()
                )
                violations.append(violation)
        
        return violations
    
    def _extract_keyword_context(self, text: str, keyword: str, context_words: int = 10) -> str:
        """Extract context around a keyword in text"""
        if not text or not keyword:
            return ""
        
        import re
        
        # Find all occurrences of the keyword (case insensitive)
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        matches = list(pattern.finditer(text))
        
        if not matches:
            return ""
        
        # Get context for the first occurrence
        match = matches[0]
        start_pos = match.start()
        
        # Split text into words
        words = text.split()
        word_positions = []
        
        # Find word boundaries
        current_pos = 0
        for word in words:
            word_start = text.find(word, current_pos)
            word_end = word_start + len(word)
            word_positions.append((word_start, word_end, word))
            current_pos = word_end
        
        # Find the word containing our keyword
        keyword_word_index = -1
        for i, (word_start, word_end, word) in enumerate(word_positions):
            if word_start <= start_pos <= word_end:
                keyword_word_index = i
                break
        
        if keyword_word_index == -1:
            return keyword  # Fallback
        
        # Extract context words
        start_word = max(0, keyword_word_index - context_words)
        end_word = min(len(words), keyword_word_index + context_words + 1)
        
        context_words_list = words[start_word:end_word]
        return " ".join(context_words_list)
    
    def _calculate_overall_risk(self, vision_results: List[VisionAnalysisResult],
                              text_results: List[TextAnalysisResult],
                              violations: List[ViolationReport]) -> Tuple[str, float]:
        """Calculate overall risk level for the document"""
        if not violations:
            return "low", 0.1
        
        # Count violations by severity
        high_violations = len([v for v in violations if v.severity == "high"])
        medium_violations = len([v for v in violations if v.severity == "medium"])
        low_violations = len([v for v in violations if v.severity == "low"])
        
        # Calculate weighted risk score
        total_pages = max(len(text_results), 1)
        total_images = len(vision_results)
        
        # Risk density (violations per page/image)
        violation_density = len(violations) / max(total_pages + total_images, 1)
        
        # Severity-weighted score (used for enhanced risk calculation)
        severity_score = (high_violations * 3 + medium_violations * 2 + low_violations * 1)
        
        # Confidence-weighted score
        avg_confidence = sum(v.confidence for v in violations) / len(violations)
        
        # Enhanced risk calculation for Middle Eastern cultural compliance
        cultural_high_risk_categories = ["adult_content", "revealing_clothing", "alcohol", "gambling", "intimate_situations"]
        cultural_violations = [v for v in violations if v.category in cultural_high_risk_categories]
        
        # Check for culturally sensitive violations
        cultural_risk_score = len(cultural_violations) / max(len(violations), 1)
        
        # Overall risk calculation with cultural weighting
        if high_violations > 0 and (violation_density > 0.2 or avg_confidence > 0.6 or cultural_risk_score > 0.3):
            return "high", min(0.95, max(avg_confidence, 0.8))
        elif high_violations > 0 or cultural_violations or (medium_violations > 1 and violation_density > 0.15):
            return "medium", min(0.85, max(avg_confidence, 0.7))
        elif medium_violations > 0 or violation_density > 0.05:
            return "medium", min(0.75, max(avg_confidence, 0.6))
        else:
            return "low", max(avg_confidence, 0.1)
    
    def _generate_summary_stats(self, image_results: List[ProcessedImage],
                               vision_results: List[VisionAnalysisResult],
                               text_results: List[TextAnalysisResult],
                               violations: List[ViolationReport]) -> Dict:
        """Generate comprehensive summary statistics"""
        
        # Image statistics
        image_stats = {
            "total_images": len(image_results),
            "images_with_violations": len(set(v.page_number for v in violations if v.violation_type == "image")),
            "avg_image_size": tuple(map(int, np.mean([img.size for img in image_results], axis=0))) if image_results else (0, 0),
            "image_formats": list(set(img.format for img in image_results)),
            "avg_caption_length": np.mean([len(img.caption.split()) for img in image_results]) if image_results else 0
        }
        
        # Vision analysis statistics
        vision_stats = {
            "avg_nsfw_score": np.mean([v.nsfw_score for v in vision_results]) if vision_results else 0,
            "high_risk_images": len([v for v in vision_results if v.risk_level == "high"]),
            "detected_categories": self._get_top_detected_categories(vision_results),
            "avg_confidence": np.mean([v.confidence for v in vision_results]) if vision_results else 0
        }
        
        # Text statistics
        text_stats = {
            "total_pages_with_text": len(text_results),
            "total_words": sum(t.total_words for t in text_results),
            "total_sentences": sum(t.total_sentences for t in text_results),
            "languages_detected": list(set(t.language for t in text_results if t.language != "unknown")),
            "avg_sentiment": np.mean([t.sentiment_score for t in text_results]) if text_results else 0,
            "total_risk_keywords": sum(len(t.risk_keywords) for t in text_results),
            "avg_text_quality": np.mean([t.text_quality_score for t in text_results]) if text_results else 0
        }
        
        # Violation statistics
        violation_stats = {
            "total_violations": len(violations),
            "violations_by_severity": {
                "high": len([v for v in violations if v.severity == "high"]),
                "medium": len([v for v in violations if v.severity == "medium"]),
                "low": len([v for v in violations if v.severity == "low"])
            },
            "violations_by_type": {
                "image": len([v for v in violations if v.violation_type == "image"]),
                "text": len([v for v in violations if v.violation_type == "text"])
            },
            "most_common_categories": self._get_most_common_violation_categories(violations),
            "avg_violation_confidence": np.mean([v.confidence for v in violations]) if violations else 0
        }
        
        return {
            "image_stats": image_stats,
            "vision_stats": vision_stats,
            "text_stats": text_stats,
            "violation_stats": violation_stats,
            "processing_summary": {
                "total_content_analyzed": len(image_results) + len(text_results),
                "violation_rate": len(violations) / max(len(image_results) + len(text_results), 1),
                "risk_distribution": self._calculate_risk_distribution(violations)
            }
        }
    
    def _get_top_detected_categories(self, vision_results: List[VisionAnalysisResult], top_k: int = 5) -> List[Tuple[str, float]]:
        """Get most commonly detected vision categories"""
        if not vision_results:
            return []
        
        # Aggregate category scores
        category_scores = {}
        for result in vision_results:
            for category, score in result.scene_classification.items():
                if category not in category_scores:
                    category_scores[category] = []
                category_scores[category].append(score)
        
        # Calculate average scores
        avg_scores = {cat: np.mean(scores) for cat, scores in category_scores.items()}
        
        # Return top categories
        sorted_categories = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_categories[:top_k]
    
    def _get_most_common_violation_categories(self, violations: List[ViolationReport], top_k: int = 5) -> List[Tuple[str, int]]:
        """Get most common violation categories"""
        from collections import Counter
        
        categories = [v.category for v in violations]
        category_counts = Counter(categories)
        return category_counts.most_common(top_k)
    
    def _calculate_risk_distribution(self, violations: List[ViolationReport]) -> Dict[str, float]:
        """Calculate distribution of risk across the document"""
        if not violations:
            return {"high": 0.0, "medium": 0.0, "low": 0.0}
        
        total = len(violations)
        distribution = {
            "high": len([v for v in violations if v.severity == "high"]) / total,
            "medium": len([v for v in violations if v.severity == "medium"]) / total,
            "low": len([v for v in violations if v.severity == "low"]) / total
        }
        
        return distribution

    def save_results(self, result: ModerationResult, output_path: str = None) -> str:
        """Save moderation results to JSON file"""
        if output_path is None:
            output_path = f"moderation_result_{result.document_id[:8]}.json"
        
        try:
            # Convert result to dictionary (handle dataclasses)
            result_dict = self._result_to_dict(result)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Results saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    def _result_to_dict(self, result: ModerationResult) -> Dict:
        """Convert ModerationResult to dictionary for JSON serialization"""
        try:
            # Use asdict for dataclasses, handle complex objects
            result_dict = asdict(result)
            
            # Clean up base64 images (truncate for storage)
            for img_result in result_dict["image_results"]:
                if "image_base64" in img_result and len(img_result["image_base64"]) > 1000:
                    img_result["image_base64"] = img_result["image_base64"][:1000] + "...[truncated]"
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Failed to convert result to dict: {e}")
            raise

# Utility functions
import numpy as np

def create_violation_summary(violations: List[ViolationReport]) -> Dict:
    """Create a concise summary of violations for quick review"""
    if not violations:
        return {"message": "No violations found", "summary": {}}
    
    summary = {
        "total_violations": len(violations),
        "severity_breakdown": {
            "high": len([v for v in violations if v.severity == "high"]),
            "medium": len([v for v in violations if v.severity == "medium"]), 
            "low": len([v for v in violations if v.severity == "low"])
        },
        "category_breakdown": {},
        "page_breakdown": {},
        "recommendations": []
    }
    
    # Category breakdown
    from collections import Counter
    categories = Counter([v.category for v in violations])
    summary["category_breakdown"] = dict(categories.most_common())
    
    # Page breakdown
    pages = Counter([v.page_number for v in violations])
    summary["page_breakdown"] = dict(pages.most_common())
    
    # Generate recommendations
    high_violations = [v for v in violations if v.severity == "high"]
    if high_violations:
        summary["recommendations"].append("Document contains high-risk content that requires immediate review")
    
    if len(violations) > 10:
        summary["recommendations"].append("Document has numerous violations - consider comprehensive content review")
    
    most_common_category = categories.most_common(1)[0][0] if categories else None
    if most_common_category:
        summary["recommendations"].append(f"Primary concern: {most_common_category} content")
    
    return summary

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python content_moderator.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Run content moderation
    moderator = ContentModerator()
    result = moderator.moderate_pdf(pdf_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"CONTENT MODERATION REPORT")
    print(f"{'='*60}")
    print(f"Document: {result.file_name}")
    print(f"Overall Risk Level: {result.overall_risk_level.upper()}")
    print(f"Confidence: {result.overall_confidence:.2f}")
    print(f"Total Violations: {result.total_violations}")
    print(f"Pages Analyzed: {result.total_pages}")
    print(f"Images Analyzed: {result.total_images}")
    
    if result.violations:
        print(f"\nTop Violations:")
        for i, violation in enumerate(result.violations[:5]):
            print(f"{i+1}. Page {violation.page_number}: {violation.description} ({violation.severity})")
    
    # Save results
    output_file = moderator.save_results(result)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Create violation summary
    summary = create_violation_summary(result.violations)
    print(f"\nViolation Summary:")
    print(json.dumps(summary, indent=2))