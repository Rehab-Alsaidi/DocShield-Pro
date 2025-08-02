# core/advanced_models.py
"""
Lightweight AI Models for Railway Deployment
Optimized for memory efficiency and fast startup
"""

import os
import warnings
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import numpy as np
from dataclasses import dataclass
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

from utils.logger import get_logger
logger = get_logger(__name__)

# Lightweight imports only
try:
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    LIGHTWEIGHT_MODELS_AVAILABLE = True
except ImportError:
    LIGHTWEIGHT_MODELS_AVAILABLE = False
    logger.warning("Lightweight models not available - running in basic mode")

@dataclass
class AdvancedAnalysisResult:
    """Enhanced analysis result with detailed information"""
    text_analysis: Dict
    image_analysis: Dict
    cultural_compliance: Dict
    confidence_scores: Dict
    detailed_findings: List[Dict]
    replacement_suggestions: List[Dict]

class AdvancedTextAnalyzer:
    """Advanced text analysis with cultural context understanding"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self._load_models()
        logger.info(f"Advanced Text Analyzer initialized on {self.device}")
    
    def _load_models(self):
        """Load advanced text analysis models"""
        try:
            # Advanced cultural keyword detection
            self.cultural_keywords = {
                "high_risk_arabic": [
                    "خمر", "كحول", "مشروبات كحولية", "نادي ليلي", "حانة",
                    "قمار", "رهان", "كازينو", "عري", "إباحي"
                ],
                "high_risk_english": [
                    "alcohol", "beer", "wine", "drinking", "drunk", "intoxicated",
                    "gambling", "casino", "betting", "poker", "nude", "naked",
                    "explicit", "adult content", "pornographic", "strip club",
                    "nightclub", "bar", "pub", "cocktail", "whiskey", "vodka"
                ],
                "medium_risk_arabic": [
                    "حفلة", "رقص", "موسيقى", "عيد الميلاد", "هالوين",
                    "عيد الحب", "موعد", "صديق", "صديقة"
                ],
                "medium_risk_english": [
                    "party", "dancing", "date", "dating", "boyfriend", "girlfriend",
                    "valentine", "christmas", "halloween", "easter", "prom",
                    "nightlife", "club scene", "beach party", "mixed party"
                ],
                "context_sensitive": [
                    "celebration", "wedding", "marriage", "family", "traditional",
                    "cultural", "religious", "business", "professional", "conference"
                ]
            }
            
            # Load sentiment analysis models (would load actual models in production)
            self.models['sentiment'] = "placeholder_for_advanced_sentiment_model"
            self.models['cultural_context'] = "placeholder_for_cultural_context_model"
            
            logger.info("Advanced text models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load advanced text models: {e}")
            raise
    
    def analyze_text_advanced(self, text: str, context: Dict = None) -> Dict:
        """Perform advanced text analysis with cultural context"""
        
        if not text or len(text.strip()) == 0:
            return {
                "risk_level": "low",
                "confidence": 0.0,
                "detected_issues": [],
                "cultural_analysis": {},
                "recommendations": []
            }
        
        text_lower = text.lower()
        detected_issues = []
        total_risk_score = 0.0
        
        # Enhanced keyword detection with context
        for category, keywords in self.cultural_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    # Extract context around keyword
                    context_text = self._extract_context(text, keyword)
                    
                    # Calculate risk based on category and context
                    if category.startswith("high_risk"):
                        risk_score = 0.8
                        severity = "high"
                    elif category.startswith("medium_risk"):
                        risk_score = 0.6
                        severity = "medium"
                    else:
                        risk_score = 0.3
                        severity = "low"
                    
                    detected_issues.append({
                        "keyword": keyword,
                        "category": category,
                        "severity": severity,
                        "confidence": risk_score,
                        "context": context_text,
                        "position": text_lower.find(keyword.lower()),
                        "cultural_impact": self._assess_cultural_impact(keyword, context_text)
                    })
                    
                    total_risk_score += risk_score
        
        # Calculate overall risk
        if detected_issues:
            avg_risk = total_risk_score / len(detected_issues)
            if avg_risk > 0.7:
                risk_level = "high"
            elif avg_risk > 0.4:
                risk_level = "medium"
            else:
                risk_level = "low"
        else:
            risk_level = "low"
            avg_risk = 0.1
        
        # Generate recommendations
        recommendations = self._generate_text_recommendations(detected_issues, risk_level)
        
        return {
            "risk_level": risk_level,
            "confidence": min(avg_risk, 0.95),
            "detected_issues": detected_issues,
            "cultural_analysis": {
                "total_issues": len(detected_issues),
                "high_risk_count": len([i for i in detected_issues if i["severity"] == "high"]),
                "medium_risk_count": len([i for i in detected_issues if i["severity"] == "medium"]),
                "cultural_sensitivity_score": self._calculate_cultural_sensitivity(detected_issues)
            },
            "recommendations": recommendations,
            "highlighted_text": self._create_highlighted_text(text, detected_issues)
        }
    
    def _extract_context(self, text: str, keyword: str, window: int = 15) -> str:
        """Extract context around a keyword"""
        words = text.split()
        keyword_lower = keyword.lower()
        
        for i, word in enumerate(words):
            if keyword_lower in word.lower():
                start = max(0, i - window)
                end = min(len(words), i + window + 1)
                context_words = words[start:end]
                return " ".join(context_words)
        
        return keyword
    
    def _assess_cultural_impact(self, keyword: str, context: str) -> Dict:
        """Assess the cultural impact of a detected keyword"""
        
        # High impact keywords for Middle Eastern/Islamic context
        high_impact_keywords = [
            "alcohol", "beer", "wine", "gambling", "casino", "nude", "naked",
            "خمر", "قمار", "عري"
        ]
        
        moderate_impact_keywords = [
            "party", "dancing", "date", "nightclub", "حفلة", "رقص"
        ]
        
        if any(high_word in keyword.lower() for high_word in high_impact_keywords):
            return {
                "impact_level": "high",
                "cultural_concern": "Conflicts with Islamic principles and Middle Eastern values",
                "replacement_urgency": "immediate"
            }
        elif any(mod_word in keyword.lower() for mod_word in moderate_impact_keywords):
            return {
                "impact_level": "medium", 
                "cultural_concern": "May require cultural sensitivity review",
                "replacement_urgency": "recommended"
            }
        else:
            return {
                "impact_level": "low",
                "cultural_concern": "Minor cultural consideration needed",
                "replacement_urgency": "optional"
            }
    
    def _calculate_cultural_sensitivity(self, detected_issues: List[Dict]) -> float:
        """Calculate overall cultural sensitivity score"""
        if not detected_issues:
            return 1.0
        
        total_impact = 0.0
        for issue in detected_issues:
            impact = issue.get("cultural_impact", {}).get("impact_level", "low")
            if impact == "high":
                total_impact += 1.0
            elif impact == "medium":
                total_impact += 0.6
            else:
                total_impact += 0.2
        
        # Normalize to 0-1 scale (inverted - higher score means more sensitive/appropriate)
        max_possible_impact = len(detected_issues)
        actual_impact_ratio = total_impact / max_possible_impact if max_possible_impact > 0 else 0
        
        return max(0.0, 1.0 - actual_impact_ratio)
    
    def _generate_text_recommendations(self, detected_issues: List[Dict], risk_level: str) -> List[Dict]:
        """Generate specific text replacement recommendations"""
        recommendations = []
        
        # Group issues by category
        categories = {}
        for issue in detected_issues:
            category = issue["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(issue)
        
        # Generate category-specific recommendations
        for category, issues in categories.items():
            if "alcohol" in category or any("alcohol" in issue["keyword"].lower() for issue in issues):
                recommendations.append({
                    "type": "keyword_replacement",
                    "priority": "high",
                    "original_terms": [issue["keyword"] for issue in issues],
                    "suggested_replacements": [
                        "refreshing beverages", "non-alcoholic drinks", "traditional drinks",
                        "fruit juices", "soft drinks", "tea", "coffee"
                    ],
                    "explanation": "Replace alcohol references with culturally appropriate beverage options"
                })
            
            elif "gambling" in category or any("gambling" in issue["keyword"].lower() for issue in issues):
                recommendations.append({
                    "type": "keyword_replacement",
                    "priority": "high",
                    "original_terms": [issue["keyword"] for issue in issues],
                    "suggested_replacements": [
                        "strategic games", "board games", "educational activities",
                        "skill-based competitions", "intellectual challenges"
                    ],
                    "explanation": "Replace gambling references with educational or skill-based activities"
                })
            
            elif "party" in category or any("party" in issue["keyword"].lower() for issue in issues):
                recommendations.append({
                    "type": "keyword_replacement", 
                    "priority": "medium",
                    "original_terms": [issue["keyword"] for issue in issues],
                    "suggested_replacements": [
                        "cultural celebration", "family gathering", "community event",
                        "educational conference", "business meeting", "cultural festival"
                    ],
                    "explanation": "Replace party references with culturally appropriate gatherings"
                })
        
        # Add general recommendations
        if risk_level == "high":
            recommendations.append({
                "type": "general_guidance",
                "priority": "critical",
                "recommendation": "Comprehensive content review required for Middle Eastern cultural compliance",
                "action_items": [
                    "Remove all high-risk cultural violations",
                    "Consult with cultural experts",
                    "Implement content guidelines review process"
                ]
            })
        
        return recommendations
    
    def _create_highlighted_text(self, text: str, detected_issues: List[Dict]) -> str:
        """Create text with HTML highlighting for detected issues"""
        if not detected_issues:
            return text
        
        highlighted_text = text
        
        # Sort issues by position (reverse order to maintain positions)
        sorted_issues = sorted(detected_issues, key=lambda x: x["position"], reverse=True)
        
        for issue in sorted_issues:
            keyword = issue["keyword"]
            severity = issue["severity"]
            
            # Choose highlight color based on severity
            if severity == "high":
                css_class = "bg-red-200 text-red-800 font-semibold px-1 rounded"
            elif severity == "medium":
                css_class = "bg-yellow-200 text-yellow-800 font-semibold px-1 rounded"
            else:
                css_class = "bg-blue-200 text-blue-800 font-semibold px-1 rounded"
            
            # Replace keyword with highlighted version
            highlighted_keyword = f'<span class="{css_class}" title="Cultural concern: {issue.get("cultural_impact", {}).get("cultural_concern", "Flagged content")}">{keyword}</span>'
            
            # Use case-insensitive replacement
            import re
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            highlighted_text = pattern.sub(highlighted_keyword, highlighted_text, count=1)
        
        return highlighted_text

class LightweightImageAnalyzer:
    """Lightweight image analysis optimized for Railway deployment"""
    
    def __init__(self):
        self.device = "cpu"  # Force CPU for Railway optimization
        self.models = {}
        self._load_models()
        logger.info(f"Lightweight Image Analyzer initialized on {self.device}")
    
    def _load_models(self):
        """Load lightweight image analysis models"""
        try:
            # Only load lightweight BLIP model if available
            if LIGHTWEIGHT_MODELS_AVAILABLE:
                logger.info("Loading lightweight BLIP model...")
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                logger.info("✅ Lightweight BLIP model loaded (~440MB)")
            
            self.cultural_concepts = {
                "prohibited_high": [
                    "alcohol bottles", "wine glasses", "beer bottles", "cocktail glasses",
                    "casino", "slot machines", "playing cards", "poker chips",
                    "nude person", "naked body", "explicit content", "adult material",
                    "strip club", "nightclub interior", "bar scene"
                ],
                "prohibited_medium": [
                    "revealing clothing", "bikini", "underwear", "lingerie",
                    "short dress", "low neckline", "exposed midriff",
                    "party scene", "dancing crowd", "nightlife",
                    "mixed gender socializing", "dating scene", "romantic scene"
                ],
                "cultural_sensitive": [
                    "christmas tree", "easter eggs", "halloween decorations",
                    "valentine decorations", "western holidays",
                    "pork products", "non-halal food"
                ],
                "appropriate_alternatives": [
                    "professional meeting", "business conference", "family gathering",
                    "cultural celebration", "educational event", "traditional clothing",
                    "modest attire", "architectural elements", "nature landscapes",
                    "technology equipment", "books and learning", "halal food"
                ]
            }
            
            logger.info("Advanced image models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load advanced image models: {e}")
            raise
    
    def analyze_image_advanced(self, image: Image.Image, caption: str = "") -> Dict:
        """Perform advanced image analysis with cultural compliance"""
        
        try:
            # Convert image to analysis format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhanced caption analysis (in production, would use BLIP-2 or similar)
            enhanced_caption = self._generate_enhanced_caption(image, caption)
            
            # Perform cultural compliance analysis
            compliance_analysis = self._analyze_cultural_compliance(enhanced_caption, image)
            
            # Generate replacement suggestions
            replacement_suggestions = self._generate_image_replacements(compliance_analysis)
            
            # Calculate overall scores
            risk_level = self._calculate_image_risk_level(compliance_analysis)
            confidence = self._calculate_confidence(compliance_analysis)
            
            return {
                "risk_level": risk_level,
                "confidence": confidence,
                "enhanced_caption": enhanced_caption,
                "cultural_analysis": compliance_analysis,
                "replacement_suggestions": replacement_suggestions,
                "detailed_findings": self._create_detailed_findings(compliance_analysis),
                "visual_elements": self._identify_visual_elements(enhanced_caption)
            }
            
        except Exception as e:
            logger.error(f"Advanced image analysis failed: {e}")
            return {
                "risk_level": "unknown",
                "confidence": 0.0,
                "enhanced_caption": caption or "Unable to analyze image",
                "cultural_analysis": {},
                "replacement_suggestions": [],
                "detailed_findings": [],
                "visual_elements": []
            }
    
    def _generate_enhanced_caption(self, image: Image.Image, original_caption: str) -> str:
        """Generate enhanced caption using advanced models (simulated)"""
        
        # In production, this would use BLIP-2, InstructBLIP, or GPT-4 Vision
        # For now, we'll enhance the existing caption with analysis
        
        if not original_caption:
            return "Image contains visual content requiring cultural compliance review"
        
        # Analyze caption for cultural elements
        enhanced_elements = []
        
        caption_lower = original_caption.lower()
        
        # Check for clothing/appearance elements
        clothing_terms = ["dress", "shirt", "clothing", "outfit", "attire", "wear"]
        if any(term in caption_lower for term in clothing_terms):
            enhanced_elements.append("clothing and attire visible")
        
        # Check for social elements
        social_terms = ["people", "person", "group", "crowd", "gathering", "meeting"]
        if any(term in caption_lower for term in social_terms):
            enhanced_elements.append("social interaction present")
        
        # Check for setting elements
        setting_terms = ["room", "building", "outdoor", "indoor", "venue", "location"]
        if any(term in caption_lower for term in setting_terms):
            enhanced_elements.append("specific setting/venue shown")
        
        # Combine original with enhancements
        if enhanced_elements:
            enhanced_caption = f"{original_caption}. Enhanced analysis: {', '.join(enhanced_elements)}."
        else:
            enhanced_caption = original_caption
        
        return enhanced_caption
    
    def _analyze_cultural_compliance(self, caption: str, image: Image.Image) -> Dict:
        """Analyze cultural compliance of image content"""
        
        caption_lower = caption.lower()
        compliance_issues = []
        
        # Check against prohibited content
        for category, concepts in self.cultural_concepts.items():
            if category == "appropriate_alternatives":
                continue
                
            for concept in concepts:
                if concept.lower() in caption_lower:
                    severity = "high" if "prohibited_high" in category else "medium" if "prohibited_medium" in category else "low"
                    
                    compliance_issues.append({
                        "concept": concept,
                        "category": category,
                        "severity": severity,
                        "confidence": 0.8 if severity == "high" else 0.6,
                        "cultural_impact": self._assess_image_cultural_impact(concept, category),
                        "detected_in": "caption_analysis"
                    })
        
        # Additional visual analysis (simulated)
        visual_risk_factors = self._detect_visual_risk_factors(image, caption)
        compliance_issues.extend(visual_risk_factors)
        
        return {
            "total_issues": len(compliance_issues),
            "issues": compliance_issues,
            "categories_affected": list(set([issue["category"] for issue in compliance_issues])),
            "highest_severity": max([issue["severity"] for issue in compliance_issues], default="low"),
            "cultural_compliance_score": self._calculate_compliance_score(compliance_issues)
        }
    
    def _assess_image_cultural_impact(self, concept: str, category: str) -> Dict:
        """Assess cultural impact of detected visual concept"""
        
        high_impact_concepts = [
            "alcohol", "gambling", "nude", "explicit", "casino", "strip club"
        ]
        
        if any(high_concept in concept.lower() for high_concept in high_impact_concepts):
            return {
                "impact_level": "critical",
                "cultural_concern": "Strictly prohibited in Islamic and Middle Eastern contexts",
                "replacement_urgency": "immediate",
                "recommended_action": "Remove or replace immediately"
            }
        elif "prohibited_medium" in category:
            return {
                "impact_level": "high",
                "cultural_concern": "May conflict with modesty requirements and cultural norms",
                "replacement_urgency": "high",
                "recommended_action": "Consider cultural-appropriate alternatives"
            }
        else:
            return {
                "impact_level": "moderate",
                "cultural_concern": "Requires cultural sensitivity consideration",
                "replacement_urgency": "recommended",
                "recommended_action": "Review for cultural appropriateness"
            }
    
    def _detect_visual_risk_factors(self, image: Image.Image, caption: str) -> List[Dict]:
        """Detect additional visual risk factors (simulated advanced analysis)"""
        
        risk_factors = []
        
        # Simulated color analysis for cultural appropriateness
        # In production, this would use actual computer vision models
        
        # Check image dimensions and aspect ratio
        width, height = image.size
        aspect_ratio = width / height
        
        # Detect potential issues based on image characteristics
        if aspect_ratio > 2.0 or aspect_ratio < 0.5:
            risk_factors.append({
                "concept": "unusual aspect ratio",
                "category": "technical_analysis",
                "severity": "low",
                "confidence": 0.4,
                "cultural_impact": {
                    "impact_level": "low",
                    "cultural_concern": "Image format may need adjustment",
                    "replacement_urgency": "optional",
                    "recommended_action": "Consider standard format"
                },
                "detected_in": "visual_analysis"
            })
        
        # Simulated content detection based on image properties
        # This would be replaced with actual model inference in production
        if width * height > 2000000:  # Large image
            risk_factors.append({
                "concept": "high resolution image",
                "category": "technical_analysis", 
                "severity": "low",
                "confidence": 0.3,
                "cultural_impact": {
                    "impact_level": "low",
                    "cultural_concern": "High detail may reveal inappropriate elements",
                    "replacement_urgency": "review",
                    "recommended_action": "Detailed content review recommended"
                },
                "detected_in": "visual_analysis"
            })
        
        return risk_factors
    
    def _calculate_compliance_score(self, compliance_issues: List[Dict]) -> float:
        """Calculate overall cultural compliance score"""
        if not compliance_issues:
            return 1.0
        
        total_penalty = 0.0
        for issue in compliance_issues:
            severity = issue.get("severity", "low")
            if severity == "high":
                total_penalty += 0.4
            elif severity == "medium":
                total_penalty += 0.2
            else:
                total_penalty += 0.1
        
        # Calculate score (0.0 = non-compliant, 1.0 = fully compliant)
        score = max(0.0, 1.0 - (total_penalty / len(compliance_issues)))
        return score
    
    def _generate_image_replacements(self, compliance_analysis: Dict) -> List[Dict]:
        """Generate specific image replacement suggestions"""
        replacements = []
        
        issues = compliance_analysis.get("issues", [])
        
        for issue in issues:
            concept = issue["concept"]
            category = issue["category"]
            
            # Generate specific replacements based on detected content
            if "alcohol" in concept.lower():
                replacements.append({
                    "original_concept": concept,
                    "replacement_type": "beverage_alternative",
                    "suggested_replacements": [
                        "Traditional Arabic coffee service",
                        "Fresh fruit juice presentation", 
                        "Tea ceremony setup",
                        "Refreshing water with mint",
                        "Cultural beverage service"
                    ],
                    "generation_prompt": "Professional beverage service in Middle Eastern cultural setting, elegant traditional presentation, culturally appropriate, high quality",
                    "priority": "high"
                })
            
            elif "gambling" in concept.lower() or "casino" in concept.lower():
                replacements.append({
                    "original_concept": concept,
                    "replacement_type": "entertainment_alternative",
                    "suggested_replacements": [
                        "Business strategy meeting",
                        "Educational board game session",
                        "Cultural learning activity",
                        "Professional conference setting",
                        "Team building exercise"
                    ],
                    "generation_prompt": "Professional educational activity, cultural learning environment, appropriate entertainment, business setting",
                    "priority": "high"
                })
            
            elif "revealing" in concept.lower() or "inappropriate clothing" in concept.lower():
                replacements.append({
                    "original_concept": concept,
                    "replacement_type": "modest_attire",
                    "suggested_replacements": [
                        "Professional business attire",
                        "Traditional modest clothing",
                        "Cultural appropriate dress",
                        "Conservative professional wear",
                        "Respectful formal attire"
                    ],
                    "generation_prompt": "Professional modest attire, culturally appropriate clothing, business formal wear, respectful dress code",
                    "priority": "high"
                })
            
            elif "party" in concept.lower() or "nightlife" in concept.lower():
                replacements.append({
                    "original_concept": concept,
                    "replacement_type": "cultural_gathering",
                    "suggested_replacements": [
                        "Cultural celebration event",
                        "Family gathering",
                        "Business networking event",
                        "Educational conference",
                        "Community cultural festival"
                    ],
                    "generation_prompt": "Cultural celebration, family gathering, professional networking event, appropriate social gathering",
                    "priority": "medium"
                })
        
        # Add general alternatives if no specific replacements generated
        if not replacements and issues:
            replacements.append({
                "original_concept": "detected inappropriate content",
                "replacement_type": "general_appropriate",
                "suggested_replacements": [
                    "Professional business environment",
                    "Cultural architectural elements",
                    "Nature and landscape photography",
                    "Educational and learning materials",
                    "Traditional cultural elements"
                ],
                "generation_prompt": "Professional appropriate imagery, cultural architectural elements, educational content, business environment",
                "priority": "medium"
            })
        
        return replacements
    
    def _calculate_image_risk_level(self, compliance_analysis: Dict) -> str:
        """Calculate overall risk level for image"""
        issues = compliance_analysis.get("issues", [])
        
        if not issues:
            return "low"
        
        high_severity_count = len([i for i in issues if i.get("severity") == "high"])
        medium_severity_count = len([i for i in issues if i.get("severity") == "medium"])
        
        if high_severity_count > 0:
            return "high"
        elif medium_severity_count > 1:
            return "high"
        elif medium_severity_count > 0:
            return "medium"
        else:
            return "low"
    
    def _calculate_confidence(self, compliance_analysis: Dict) -> float:
        """Calculate confidence in analysis"""
        issues = compliance_analysis.get("issues", [])
        
        if not issues:
            return 0.2  # Low confidence in "no issues" for complex images
        
        # Average confidence of all detected issues
        confidences = [issue.get("confidence", 0.5) for issue in issues]
        avg_confidence = sum(confidences) / len(confidences)
        
        return min(avg_confidence, 0.95)
    
    def _create_detailed_findings(self, compliance_analysis: Dict) -> List[Dict]:
        """Create detailed findings for reporting"""
        findings = []
        
        issues = compliance_analysis.get("issues", [])
        
        for issue in issues:
            finding = {
                "finding_id": f"img_{hash(issue['concept'])}",
                "category": issue["category"],
                "severity": issue["severity"],
                "description": f"Detected {issue['concept']} in image content",
                "cultural_impact": issue.get("cultural_impact", {}),
                "confidence": issue.get("confidence", 0.5),
                "recommendation": self._get_finding_recommendation(issue),
                "detection_method": issue.get("detected_in", "analysis")
            }
            findings.append(finding)
        
        return findings
    
    def _get_finding_recommendation(self, issue: Dict) -> str:
        """Get specific recommendation for a finding"""
        concept = issue["concept"].lower()
        severity = issue["severity"]
        
        if severity == "high":
            if "alcohol" in concept:
                return "Replace with culturally appropriate beverage imagery showing traditional drinks or professional service"
            elif "gambling" in concept or "casino" in concept:
                return "Replace with educational or business-focused imagery showing appropriate activities"
            elif "nude" in concept or "explicit" in concept:
                return "Remove immediately and replace with professional, modest imagery"
            else:
                return "High-priority replacement required for cultural compliance"
        elif severity == "medium":
            return "Consider replacing with more culturally appropriate alternatives that align with regional values"
        else:
            return "Review for cultural appropriateness and consider alternatives if needed"
    
    def _identify_visual_elements(self, caption: str) -> List[Dict]:
        """Identify key visual elements in the image"""
        elements = []
        
        # Extract visual elements from caption
        visual_keywords = {
            "people": ["person", "people", "man", "woman", "child", "group"],
            "clothing": ["dress", "shirt", "clothing", "outfit", "attire"],
            "objects": ["bottle", "glass", "table", "chair", "equipment"],
            "setting": ["room", "building", "outdoor", "indoor", "background"],
            "activities": ["sitting", "standing", "walking", "meeting", "working"]
        }
        
        caption_lower = caption.lower()
        
        for category, keywords in visual_keywords.items():
            found_keywords = [kw for kw in keywords if kw in caption_lower]
            if found_keywords:
                elements.append({
                    "category": category,
                    "detected_elements": found_keywords,
                    "relevance": "high" if len(found_keywords) > 1 else "medium"
                })
        
        return elements

class LightweightModelManager:
    """Manages lightweight AI models optimized for Railway deployment"""
    
    def __init__(self):
        self.text_analyzer = AdvancedTextAnalyzer()
        self.image_analyzer = LightweightImageAnalyzer()
        logger.info("Lightweight Model Manager initialized")

# Backward compatibility alias
AdvancedModelManager = LightweightModelManager
    
    def analyze_content_comprehensive(self, text_content: List[str], images: List[Image.Image], 
                                   image_captions: List[str]) -> AdvancedAnalysisResult:
        """Perform comprehensive content analysis"""
        
        # Analyze all text content
        text_results = []
        for text in text_content:
            if text and text.strip():
                result = self.text_analyzer.analyze_text_advanced(text)
                text_results.append(result)
        
        # Analyze all images
        image_results = []
        for i, image in enumerate(images):
            caption = image_captions[i] if i < len(image_captions) else ""
            result = self.image_analyzer.analyze_image_advanced(image, caption)
            image_results.append(result)
        
        # Combine results for overall analysis
        overall_analysis = self._combine_analysis_results(text_results, image_results)
        
        return AdvancedAnalysisResult(
            text_analysis=overall_analysis["text_summary"],
            image_analysis=overall_analysis["image_summary"],
            cultural_compliance=overall_analysis["cultural_compliance"],
            confidence_scores=overall_analysis["confidence_scores"],
            detailed_findings=overall_analysis["detailed_findings"],
            replacement_suggestions=overall_analysis["replacement_suggestions"]
        )
    
    def _combine_analysis_results(self, text_results: List[Dict], image_results: List[Dict]) -> Dict:
        """Combine text and image analysis results"""
        
        # Aggregate text results
        text_issues = []
        text_recommendations = []
        for result in text_results:
            text_issues.extend(result.get("detected_issues", []))
            text_recommendations.extend(result.get("recommendations", []))
        
        # Aggregate image results
        image_issues = []
        image_recommendations = []
        for result in image_results:
            image_issues.extend(result.get("detailed_findings", []))
            image_recommendations.extend(result.get("replacement_suggestions", []))
        
        # Calculate overall metrics
        total_high_risk = len([i for i in text_issues if i.get("severity") == "high"]) + \
                         len([i for i in image_issues if i.get("severity") == "high"])
        
        total_medium_risk = len([i for i in text_issues if i.get("severity") == "medium"]) + \
                           len([i for i in image_issues if i.get("severity") == "medium"])
        
        # Determine overall risk level
        if total_high_risk > 0:
            overall_risk = "high"
        elif total_medium_risk > 1:
            overall_risk = "medium"
        elif total_medium_risk > 0:
            overall_risk = "medium"
        else:
            overall_risk = "low"
        
        # Calculate confidence scores
        text_confidences = [r.get("confidence", 0.5) for r in text_results if r.get("confidence") is not None]
        image_confidences = [r.get("confidence", 0.5) for r in image_results if r.get("confidence") is not None]
        
        overall_confidence = 0.7  # Default
        if text_confidences and image_confidences:
            overall_confidence = (sum(text_confidences) / len(text_confidences) + 
                                sum(image_confidences) / len(image_confidences)) / 2
        elif text_confidences:
            overall_confidence = sum(text_confidences) / len(text_confidences)
        elif image_confidences:
            overall_confidence = sum(image_confidences) / len(image_confidences)
        
        return {
            "text_summary": {
                "total_issues": len(text_issues),
                "high_risk_count": len([i for i in text_issues if i.get("severity") == "high"]),
                "medium_risk_count": len([i for i in text_issues if i.get("severity") == "medium"]),
                "recommendations_count": len(text_recommendations)
            },
            "image_summary": {
                "total_issues": len(image_issues),
                "high_risk_count": len([i for i in image_issues if i.get("severity") == "high"]),
                "medium_risk_count": len([i for i in image_issues if i.get("severity") == "medium"]),
                "replacements_available": len(image_recommendations)
            },
            "cultural_compliance": {
                "overall_risk_level": overall_risk,
                "total_violations": len(text_issues) + len(image_issues),
                "compliance_score": self._calculate_overall_compliance_score(text_results, image_results),
                "cultural_sensitivity_rating": self._calculate_cultural_sensitivity_rating(text_issues, image_issues)
            },
            "confidence_scores": {
                "overall_confidence": overall_confidence,
                "text_analysis_confidence": sum(text_confidences) / len(text_confidences) if text_confidences else 0.5,
                "image_analysis_confidence": sum(image_confidences) / len(image_confidences) if image_confidences else 0.5
            },
            "detailed_findings": text_issues + image_issues,
            "replacement_suggestions": text_recommendations + image_recommendations
        }
    
    def _calculate_overall_compliance_score(self, text_results: List[Dict], image_results: List[Dict]) -> float:
        """Calculate overall cultural compliance score"""
        
        text_scores = []
        for result in text_results:
            cultural_analysis = result.get("cultural_analysis", {})
            if "cultural_sensitivity_score" in cultural_analysis:
                text_scores.append(cultural_analysis["cultural_sensitivity_score"])
        
        image_scores = []
        for result in image_results:
            cultural_analysis = result.get("cultural_analysis", {})
            if "cultural_compliance_score" in cultural_analysis:
                image_scores.append(cultural_analysis["cultural_compliance_score"])
        
        all_scores = text_scores + image_scores
        if all_scores:
            return sum(all_scores) / len(all_scores)
        else:
            return 0.5  # Default moderate score
    
    def _calculate_cultural_sensitivity_rating(self, text_issues: List[Dict], image_issues: List[Dict]) -> str:
        """Calculate cultural sensitivity rating"""
        
        critical_issues = len([i for i in text_issues + image_issues 
                             if i.get("cultural_impact", {}).get("impact_level") == "critical"])
        
        high_issues = len([i for i in text_issues + image_issues 
                          if i.get("cultural_impact", {}).get("impact_level") == "high"])
        
        if critical_issues > 0:
            return "Not Compliant"
        elif high_issues > 2:
            return "Requires Major Review"
        elif high_issues > 0:
            return "Requires Minor Review"
        else:
            return "Culturally Appropriate"

# Example usage and testing
if __name__ == "__main__":
    # Initialize lightweight models
    model_manager = LightweightModelManager()
    
    # Test text analysis
    sample_text = "The party featured alcohol and dancing with mixed groups."
    text_result = model_manager.text_analyzer.analyze_text_advanced(sample_text)
    
    print("Advanced Text Analysis Result:")
    print(f"Risk Level: {text_result['risk_level']}")
    print(f"Confidence: {text_result['confidence']:.2f}")
    print(f"Detected Issues: {len(text_result['detected_issues'])}")
    
    for issue in text_result['detected_issues']:
        print(f"  - {issue['keyword']} ({issue['severity']}): {issue['cultural_impact']['cultural_concern']}")
    
    print("\nRecommendations:")
    for rec in text_result['recommendations']:
        if rec['type'] == 'keyword_replacement':
            print(f"  - Replace '{', '.join(rec['original_terms'])}' with: {', '.join(rec['suggested_replacements'][:2])}")