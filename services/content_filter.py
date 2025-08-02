# services/content_filter.py
from typing import Dict, List, Tuple, Optional, Set
import re
from dataclasses import dataclass
from collections import defaultdict

from utils.logger import get_logger
from app.config import CONTENT_RULES
from utils.helpers import clean_text

logger = get_logger(__name__)

@dataclass
class FilterResult:
    """Result from content filtering"""
    is_violation: bool
    severity: str
    confidence: float
    matched_rules: List[Dict]
    categories: List[str]
    risk_score: float

class ContentFilter:
    """Advanced content filtering with customizable rules"""
    
    def __init__(self):
        self.rules = CONTENT_RULES
        self.compiled_patterns = self._compile_regex_patterns()
        self.category_weights = {
            'high': 3.0,
            'medium': 2.0, 
            'low': 1.0
        }
        logger.info("Content Filter initialized")
    
    def _compile_regex_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for faster matching"""
        compiled = defaultdict(list)
        
        for risk_level, categories in self.rules.items():
            for category, rules in categories.items():
                for keyword in rules['keywords']:
                    # Create word boundary patterns
                    pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                    try:
                        compiled_pattern = re.compile(pattern, re.IGNORECASE)
                        compiled[category].append({
                            'pattern': compiled_pattern,
                            'keyword': keyword,
                            'severity': rules['severity'],
                            'confidence_threshold': rules['confidence_threshold']
                        })
                    except re.error as e:
                        logger.warning(f"Failed to compile regex for keyword '{keyword}': {e}")
        
        return compiled
    
    def filter_text(self, text: str, custom_threshold: float = None) -> FilterResult:
        """Filter text content for violations"""
        if not text or len(text.strip()) < 3:
            return FilterResult(
                is_violation=False,
                severity='low',
                confidence=0.0,
                matched_rules=[],
                categories=[],
                risk_score=0.0
            )
        
        # Clean and normalize text
        cleaned_text = clean_text(text)
        text_lower = cleaned_text.lower()
        
        matched_rules = []
        categories = set()
        total_risk_score = 0.0
        
        # Check against compiled patterns
        for category, patterns in self.compiled_patterns.items():
            for pattern_info in patterns:
                matches = list(pattern_info['pattern'].finditer(text_lower))
                
                if matches:
                    match_count = len(matches)
                    confidence = min(0.95, pattern_info['confidence_threshold'] + (match_count - 1) * 0.1)
                    
                    matched_rule = {
                        'keyword': pattern_info['keyword'],
                        'category': category,
                        'severity': pattern_info['severity'],
                        'confidence': confidence,
                        'match_count': match_count,
                        'positions': [(m.start(), m.end()) for m in matches]
                    }
                    
                    matched_rules.append(matched_rule)
                    categories.add(category)
                    
                    # Calculate risk score
                    severity_weight = self.category_weights.get(pattern_info['severity'], 1.0)
                    total_risk_score += confidence * severity_weight * match_count
        
        # Advanced pattern matching
        advanced_matches = self._advanced_pattern_matching(cleaned_text)
        matched_rules.extend(advanced_matches)
        
        # Determine overall result
        is_violation = len(matched_rules) > 0
        overall_severity = self._calculate_overall_severity(matched_rules)
        overall_confidence = self._calculate_overall_confidence(matched_rules)
        
        # Apply custom threshold if provided
        if custom_threshold and overall_confidence < custom_threshold:
            is_violation = False
        
        return FilterResult(
            is_violation=is_violation,
            severity=overall_severity,
            confidence=overall_confidence,
            matched_rules=matched_rules,
            categories=list(categories),
            risk_score=min(10.0, total_risk_score)
        )
    
    def _advanced_pattern_matching(self, text: str) -> List[Dict]:
        """Advanced pattern matching for complex content detection"""
        advanced_matches = []
        text_lower = text.lower()
        
        # URL pattern matching
        url_pattern = re.compile(r'https?://[^\s]+', re.IGNORECASE)
        urls = url_pattern.findall(text)
        
        for url in urls:
            if self._is_suspicious_url(url):
                advanced_matches.append({
                    'keyword': 'suspicious_url',
                    'category': 'external_links',
                    'severity': 'medium',
                    'confidence': 0.7,
                    'match_count': 1,
                    'positions': []
                })
        
        # Email pattern matching
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        emails = email_pattern.findall(text)
        
        if len(emails) > 5:  # Many emails might indicate spam
            advanced_matches.append({
                'keyword': 'multiple_emails',
                'category': 'contact_information',
                'severity': 'low',
                'confidence': 0.6,
                'match_count': len(emails),
                'positions': []
            })
        
        # Phone number pattern
        phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        phones = phone_pattern.findall(text)
        
        if len(phones) > 3:
            advanced_matches.append({
                'keyword': 'multiple_phone_numbers',
                'category': 'contact_information',
                'severity': 'low',
                'confidence': 0.5,
                'match_count': len(phones),
                'positions': []
            })
        
        # Repeated text detection (potential spam)
        if self._detect_repeated_content(text_lower):
            advanced_matches.append({
                'keyword': 'repeated_content',
                'category': 'spam_indicators',
                'severity': 'medium',
                'confidence': 0.8,
                'match_count': 1,
                'positions': []
            })
        
        # Excessive capitalization
        if self._detect_excessive_caps(text):
            advanced_matches.append({
                'keyword': 'excessive_capitalization',
                'category': 'formatting_issues',
                'severity': 'low',
                'confidence': 0.4,
                'match_count': 1,
                'positions': []
            })
        
        return advanced_matches
    
    def _is_suspicious_url(self, url: str) -> bool:
        """Check if URL is suspicious"""
        suspicious_domains = [
            'bit.ly', 'tinyurl.com', 't.co', 'goo.gl',
            'ow.ly', 'short.link'
        ]
        
        suspicious_keywords = [
            'download', 'free', 'win', 'prize', 'click',
            'urgent', 'limited', 'offer'
        ]
        
        url_lower = url.lower()
        
        # Check for suspicious domains
        for domain in suspicious_domains:
            if domain in url_lower:
                return True
        
        # Check for suspicious keywords in URL
        for keyword in suspicious_keywords:
            if keyword in url_lower:
                return True
        
        return False
    
    def _detect_repeated_content(self, text: str) -> bool:
        """Detect if content has excessive repetition"""
        words = text.split()
        
        if len(words) < 10:
            return False
        
        # Check for repeated phrases
        phrases = []
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            phrases.append(phrase)
        
        # Count phrase frequency
        phrase_counts = {}
        for phrase in phrases:
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        
        # Check if any phrase repeats too often
        max_repetitions = max(phrase_counts.values()) if phrase_counts else 0
        repetition_ratio = max_repetitions / len(phrases) if phrases else 0
        
        return repetition_ratio > 0.3  # More than 30% repetition
    
    def _detect_excessive_caps(self, text: str) -> bool:
        """Detect excessive capitalization"""
        if len(text) < 20:
            return False
        
        caps_count = sum(1 for c in text if c.isupper())
        caps_ratio = caps_count / len(text)
        
        return caps_ratio > 0.5  # More than 50% capitals
    
    def _calculate_overall_severity(self, matched_rules: List[Dict]) -> str:
        """Calculate overall severity from matched rules"""
        if not matched_rules:
            return 'low'
        
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        
        for rule in matched_rules:
            severity = rule.get('severity', 'low')
            if severity in severity_counts:
                severity_counts[severity] += rule.get('match_count', 1)
        
        # Determine overall severity
        if severity_counts['high'] > 0:
            return 'high'
        elif severity_counts['medium'] > 2:
            return 'high'
        elif severity_counts['medium'] > 0:
            return 'medium'
        elif severity_counts['low'] > 5:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_overall_confidence(self, matched_rules: List[Dict]) -> float:
        """Calculate overall confidence from matched rules"""
        if not matched_rules:
            return 0.0
        
        # Weight by severity and match count
        weighted_confidences = []
        
        for rule in matched_rules:
            confidence = rule.get('confidence', 0.0)
            severity = rule.get('severity', 'low')
            match_count = rule.get('match_count', 1)
            
            weight = self.category_weights.get(severity, 1.0)
            weighted_confidence = confidence * weight * min(match_count, 3)  # Cap at 3 matches
            weighted_confidences.append(weighted_confidence)
        
        # Calculate weighted average
        if weighted_confidences:
            overall_confidence = sum(weighted_confidences) / len(weighted_confidences)
            return min(0.95, overall_confidence / 3.0)  # Normalize
        
        return 0.0
    
    def filter_image_caption(self, caption: str, image_metadata: Dict = None) -> FilterResult:
        """Filter image caption with additional context"""
        # Base text filtering
        result = self.filter_text(caption)
        
        # Add image-specific rules
        if image_metadata:
            image_matches = self._check_image_specific_rules(caption, image_metadata)
            result.matched_rules.extend(image_matches)
            
            # Recalculate if new matches found
            if image_matches:
                result.is_violation = True
                result.severity = self._calculate_overall_severity(result.matched_rules)
                result.confidence = self._calculate_overall_confidence(result.matched_rules)
        
        return result
    
    def _check_image_specific_rules(self, caption: str, metadata: Dict) -> List[Dict]:
        """Check image-specific filtering rules"""
        matches = []
        caption_lower = caption.lower()
        
        # Size-based rules
        image_size = metadata.get('size', (0, 0))
        if max(image_size) > 2000:  # Large images might be high-resolution inappropriate content
            if any(keyword in caption_lower for keyword in ['person', 'people', 'body', 'skin']):
                matches.append({
                    'keyword': 'high_res_person_image',
                    'category': 'image_analysis',
                    'severity': 'medium',
                    'confidence': 0.6,
                    'match_count': 1,
                    'positions': []
                })
        
        # Content-based rules
        revealing_indicators = ['revealing', 'exposed', 'bare', 'minimal clothing', 'undressed']
        if any(indicator in caption_lower for indicator in revealing_indicators):
            matches.append({
                'keyword': 'revealing_content_detected',
                'category': 'inappropriate_content',
                'severity': 'high',
                'confidence': 0.8,
                'match_count': 1,
                'positions': []
            })
        
        return matches
    
    def batch_filter(self, texts: List[str], batch_size: int = 100) -> List[FilterResult]:
        """Filter multiple texts in batches"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch:
                result = self.filter_text(text)
                batch_results.append(result)
            
            results.extend(batch_results)
            logger.debug(f"Processed batch {i//batch_size + 1}, filtered {len(batch)} texts")
        
        return results
    
    def create_custom_rule(self, category: str, keywords: List[str], 
                          severity: str = 'medium', confidence: float = 0.7) -> bool:
        """Add custom filtering rule"""
        try:
            if category not in self.compiled_patterns:
                self.compiled_patterns[category] = []
            
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                compiled_pattern = re.compile(pattern, re.IGNORECASE)
                
                self.compiled_patterns[category].append({
                    'pattern': compiled_pattern,
                    'keyword': keyword,
                    'severity': severity,
                    'confidence_threshold': confidence
                })
            
            logger.info(f"Added custom rule for category '{category}' with {len(keywords)} keywords")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create custom rule: {e}")
            return False
    
    def get_filter_statistics(self) -> Dict:
        """Get filtering statistics"""
        stats = {
            'total_categories': len(self.compiled_patterns),
            'total_patterns': sum(len(patterns) for patterns in self.compiled_patterns.values()),
            'categories': {}
        }
        
        for category, patterns in self.compiled_patterns.items():
            stats['categories'][category] = {
                'pattern_count': len(patterns),
                'severity_distribution': {}
            }
            
            # Count by severity
            for pattern_info in patterns:
                severity = pattern_info['severity']
                if severity not in stats['categories'][category]['severity_distribution']:
                    stats['categories'][category]['severity_distribution'][severity] = 0
                stats['categories'][category]['severity_distribution'][severity] += 1
        
        return stats
    
    def export_rules(self) -> Dict:
        """Export current filtering rules"""
        exported_rules = {}
        
        for category, patterns in self.compiled_patterns.items():
            exported_rules[category] = []
            
            for pattern_info in patterns:
                exported_rules[category].append({
                    'keyword': pattern_info['keyword'],
                    'severity': pattern_info['severity'],
                    'confidence_threshold': pattern_info['confidence_threshold']
                })
        
        return exported_rules
    
    def import_rules(self, rules: Dict) -> bool:
        """Import filtering rules"""
        try:
            self.compiled_patterns.clear()
            
            for category, rule_list in rules.items():
                self.compiled_patterns[category] = []
                
                for rule in rule_list:
                    keyword = rule['keyword']
                    pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                    compiled_pattern = re.compile(pattern, re.IGNORECASE)
                    
                    self.compiled_patterns[category].append({
                        'pattern': compiled_pattern,
                        'keyword': keyword,
                        'severity': rule.get('severity', 'medium'),
                        'confidence_threshold': rule.get('confidence_threshold', 0.7)
                    })
            
            logger.info("Successfully imported filtering rules")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import rules: {e}")
            return False

# Utility functions for content filtering

def quick_text_filter(text: str, severity_threshold: str = 'medium') -> bool:
    """Quick check if text contains violations above threshold"""
    filter_instance = ContentFilter()
    result = filter_instance.filter_text(text)
    
    severity_levels = {'low': 1, 'medium': 2, 'high': 3}
    threshold_level = severity_levels.get(severity_threshold, 2)
    result_level = severity_levels.get(result.severity, 1)
    
    return result.is_violation and result_level >= threshold_level

def extract_violation_context(text: str, matched_rules: List[Dict], context_chars: int = 50) -> List[Dict]:
    """Extract context around violations for review"""
    contexts = []
    
    for rule in matched_rules:
        positions = rule.get('positions', [])
        
        for start, end in positions:
            context_start = max(0, start - context_chars)
            context_end = min(len(text), end + context_chars)
            
            context = {
                'keyword': rule['keyword'],
                'category': rule['category'],
                'severity': rule['severity'],
                'context': text[context_start:context_end],
                'position': (start, end),
                'highlighted_text': text[start:end]
            }
            
            contexts.append(context)
    
    return contexts