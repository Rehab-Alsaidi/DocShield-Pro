# core/nlp_analyzer.py
import fitz  # PyMuPDF for text extraction
import re

# Basic NLP without heavy ML dependencies
print("📄 Using lightweight text analysis")
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from collections import Counter
# Optional NLP libraries with graceful fallbacks
SPACY_AVAILABLE = False
spacy = None

try:
    import spacy
    SPACY_AVAILABLE = True
    print("✅ Spacy imported successfully")
except Exception as e:
    print(f"⚠️ Spacy not available: {e}")
    SPACY_AVAILABLE = False

from utils.logger import get_logger

# Simple config without dependencies
class SimpleConfig:
    def __init__(self):
        self.device = "cpu"

model_config = SimpleConfig()
print("✅ Simple config loaded")

CONTENT_RULES = {"high_risk": {"adult_content": {"keywords": ["explicit"], "severity": "high"}}}

logger = get_logger(__name__)

@dataclass
class TextAnalysisResult:
    """Result from text analysis"""
    total_words: int
    total_sentences: int
    language: str
    sentiment_score: float
    risk_keywords: List[Dict]
    content_categories: Dict[str, float]
    semantic_similarities: Dict[str, float]
    risk_level: str
    confidence: float
    extracted_entities: List[Dict]
    text_quality_score: float
    analysis_details: Dict

@dataclass
class ExtractedText:
    """Structure for extracted text from PDF"""
    page_number: int
    text_content: str
    font_info: List[Dict]
    text_position: List[Tuple[float, float, float, float]]  # bbox coordinates
    text_type: str  # 'body', 'header', 'footer', 'title'

class TextExtractor:
    """Extract and preprocess text from PDF"""
    
    def __init__(self):
        self.min_text_length = 10
        logger.info("Text Extractor initialized")
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[ExtractedText]:
        """Extract structured text from PDF"""
        extracted_texts = []
        
        try:
            pdf_document = fitz.open(pdf_path)
            logger.info(f"Extracting text from PDF with {pdf_document.page_count} pages")
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                
                # Extract text blocks with position and font information
                text_blocks = page.get_text("dict")
                
                page_text = ""
                font_info = []
                text_positions = []
                
                for block in text_blocks["blocks"]:
                    if "lines" in block:  # Text block
                        for line in block["lines"]:
                            line_text = ""
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text:
                                    line_text += text + " "
                                    
                                    # Collect font information
                                    font_info.append({
                                        "text": text,
                                        "font": span["font"],
                                        "size": span["size"],
                                        "flags": span["flags"],
                                        "color": span["color"]
                                    })
                                    
                                    # Collect position information
                                    bbox = span["bbox"]
                                    text_positions.append(bbox)
                            
                            if line_text.strip():
                                page_text += line_text.strip() + "\n"
                
                # Classify text type based on position and font
                text_type = self._classify_text_type(text_positions, font_info, page.rect)
                
                if len(page_text.strip()) >= self.min_text_length:
                    extracted_texts.append(ExtractedText(
                        page_number=page_num + 1,
                        text_content=page_text.strip(),
                        font_info=font_info,
                        text_position=text_positions,
                        text_type=text_type
                    ))
            
            pdf_document.close()
            logger.info(f"Extracted text from {len(extracted_texts)} pages")
            return extracted_texts
            
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            return []
    
    def _classify_text_type(self, positions: List[Tuple], font_info: List[Dict], page_rect) -> str:
        """Classify text as header, footer, title, or body"""
        if not positions or not font_info:
            return "body"
        
        # Analyze vertical positions
        y_positions = [pos[1] for pos in positions]  # Top y-coordinates
        page_height = page_rect.height
        
        # Check if text is in header area (top 15% of page)
        header_threshold = page_height * 0.15
        footer_threshold = page_height * 0.85
        
        avg_y = np.mean(y_positions)
        
        if avg_y < header_threshold:
            return "header"
        elif avg_y > footer_threshold:
            return "footer"
        
        # Check for title (larger font size)
        avg_font_size = np.mean([font["size"] for font in font_info])
        if avg_font_size > 16:  # Typically titles have larger fonts
            return "title"
        
        return "body"

class SemanticAnalyzer:
    """Semantic analysis using sentence transformers"""
    
    def __init__(self):
        self.device = model_config.device
        self.model = None
        self._load_model()
        
        # Predefined content category descriptions
        self.content_categories = {
            "religious_content": "Religious content, worship, prayer, sacred texts, religious symbols",
            "adult_content": "Adult content, sexual themes, intimate relationships, mature themes",
            "alcohol_drugs": "Alcohol consumption, drinking, bars, drugs, substance use",
            "gambling": "Gambling, betting, casinos, poker, lottery, games of chance",
            "violence": "Violence, fighting, weapons, aggression, conflict, war",
            "dating_romance": "Dating, romantic relationships, love, marriage, couples",
            "western_culture": "Western holidays, Christmas, Easter, Halloween, Valentine's Day",
            "business_finance": "Business, finance, money, investment, corporate activities",
            "education": "Education, learning, schools, academic content, studying",
            "family_children": "Family activities, children, parenting, family relationships",
            "food_dining": "Food, restaurants, cooking, dining, recipes",
            "technology": "Technology, computers, internet, digital devices, software",
            "travel_tourism": "Travel, tourism, vacation, destinations, hotels",
            "health_medical": "Health, medical content, healthcare, wellness, fitness",
            "politics_government": "Politics, government, elections, policy, political figures"
        }
    
    def _load_model(self):
        """Load sentence transformer model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("Sentence Transformers not available. Semantic analysis will be disabled.")
            self.model = None
            return
            
        try:
            logger.info("Loading Sentence Transformer model...")
            
            model_name = model_config.sentence_transformer
            self.model = SentenceTransformer(model_name, device=self.device)
            
            logger.info("Sentence Transformer model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer model: {e}")
            self.model = None
    
    def analyze_semantic_content(self, text: str) -> Dict[str, float]:
        """Analyze semantic content categories"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE or self.model is None:
            logger.debug("Semantic analysis disabled - returning empty results")
            return {}
            
        try:
            if not text.strip():
                return {}
            
            # Create embeddings for text and categories
            text_embedding = self.model.encode([text])
            category_embeddings = self.model.encode(list(self.content_categories.values()))
            
            # Calculate similarities
            similarities = cosine_similarity(text_embedding, category_embeddings)[0]
            
            # Create category-similarity mapping
            semantic_scores = {}
            for i, (category, _) in enumerate(self.content_categories.items()):
                semantic_scores[category] = float(similarities[i])
            
            return semantic_scores
            
        except Exception as e:
            logger.error(f"Semantic analysis failed: {e}")
            return {}

class KeywordAnalyzer:
    """Keyword-based content analysis"""
    
    def __init__(self):
        self.content_rules = CONTENT_RULES
        logger.info("Keyword Analyzer initialized")
    
    def analyze_keywords(self, text: str) -> Tuple[List[Dict], Dict[str, int]]:
        """Analyze text for risk keywords"""
        if not text:
            return [], {}
        
        text_lower = text.lower()
        found_keywords = []
        category_counts = {}
        
        for risk_level, categories in self.content_rules.items():
            for category, rules in categories.items():
                category_count = 0
                
                for keyword in rules["keywords"]:
                    # Use word boundaries for better matching
                    pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                    matches = re.findall(pattern, text_lower)
                    
                    if matches:
                        found_keywords.append({
                            "keyword": keyword,
                            "category": category,
                            "risk_level": risk_level,
                            "severity": rules["severity"],
                            "count": len(matches),
                            "confidence": rules["confidence_threshold"]
                        })
                        category_count += len(matches)
                
                if category_count > 0:
                    category_counts[category] = category_count
        
        return found_keywords, category_counts
    
    def calculate_keyword_risk_score(self, found_keywords: List[Dict]) -> Tuple[str, float]:
        """Calculate overall risk score based on keywords"""
        if not found_keywords:
            return "low", 0.1
        
        # Weight by severity and count
        severity_weights = {"high": 1.0, "medium": 0.6, "low": 0.3}
        total_score = 0.0
        max_severity = "low"
        
        for keyword_info in found_keywords:
            severity = keyword_info["severity"]
            count = keyword_info["count"]
            weight = severity_weights.get(severity, 0.3)
            
            total_score += weight * count * keyword_info["confidence"]
            
            if severity == "high":
                max_severity = "high"
            elif severity == "medium" and max_severity != "high":
                max_severity = "medium"
        
        # Normalize score
        confidence = min(0.95, total_score / 10.0)
        
        # Determine risk level
        if total_score > 2.0 or max_severity == "high":
            return "high", confidence
        elif total_score > 0.8 or max_severity == "medium":
            return "medium", confidence
        else:
            return "low", confidence

class EntityExtractor:
    """Named Entity Recognition for additional context"""
    
    def __init__(self):
        self.nlp = None
        self._load_spacy_model()
    
    def _load_spacy_model(self):
        """Load spaCy model for NER"""
        if not SPACY_AVAILABLE:
            logger.warning("SpaCy not available. NER will be disabled.")
            self.nlp = None
            return
            
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy NER model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SpaCy model: {e}. NER will be disabled.")
            self.nlp = None
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities from text"""
        if not self.nlp or not text:
            return []
        
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "description": spacy.explain(ent.label_),
                    "start": ent.start_char,
                    "end": ent.end_char
                })
            
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []

class NLPAnalyzer:
    """Main NLP analyzer combining all text analysis components"""
    
    def __init__(self):
        self.text_extractor = TextExtractor()
        self.semantic_analyzer = SemanticAnalyzer()
        self.keyword_analyzer = KeywordAnalyzer()
        self.entity_extractor = EntityExtractor()
        logger.info("NLP Analyzer initialized")
    
    def analyze_text(self, text: str) -> TextAnalysisResult:
        """Comprehensive text analysis"""
        try:
            if not text or len(text.strip()) < 10:
                return self._create_empty_result()
            
            logger.debug(f"Analyzing text of length: {len(text)}")
            
            # Basic text statistics
            words = text.split()
            sentences = re.split(r'[.!?]+', text)
            word_count = len(words)
            sentence_count = len([s for s in sentences if s.strip()])
            
            # Language detection and sentiment
            try:
                if TEXTBLOB_AVAILABLE:
                    blob = TextBlob(text)
                    language = blob.detect_language()
                    sentiment_score = blob.sentiment.polarity
                else:
                    language = "unknown"
                    sentiment_score = 0.0
            except:
                language = "unknown"
                sentiment_score = 0.0
            
            # Keyword analysis
            risk_keywords, category_counts = self.keyword_analyzer.analyze_keywords(text)
            keyword_risk_level, keyword_confidence = self.keyword_analyzer.calculate_keyword_risk_score(risk_keywords)
            
            # Semantic analysis
            semantic_similarities = self.semantic_analyzer.analyze_semantic_content(text)
            
            # Entity extraction
            entities = self.entity_extractor.extract_entities(text)
            
            # Content categorization
            content_categories = self._categorize_content(semantic_similarities, category_counts)
            
            # Overall risk assessment
            overall_risk_level, overall_confidence = self._calculate_overall_risk(
                keyword_risk_level, keyword_confidence, semantic_similarities, risk_keywords
            )
            
            # Text quality score
            quality_score = self._calculate_text_quality(text, words, sentences)
            
            # Analysis details
            analysis_details = {
                "word_frequency": self._get_word_frequency(words),
                "sentence_lengths": [len(s.split()) for s in sentences if s.strip()],
                "category_distribution": category_counts,
                "semantic_top_categories": self._get_top_semantic_categories(semantic_similarities),
                "risk_distribution": self._analyze_risk_distribution(risk_keywords),
                "text_complexity": self._analyze_text_complexity(text)
            }
            
            return TextAnalysisResult(
                total_words=word_count,
                total_sentences=sentence_count,
                language=language,
                sentiment_score=sentiment_score,
                risk_keywords=risk_keywords,
                content_categories=content_categories,
                semantic_similarities=semantic_similarities,
                risk_level=overall_risk_level,
                confidence=overall_confidence,
                extracted_entities=entities,
                text_quality_score=quality_score,
                analysis_details=analysis_details
            )
            
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return self._create_empty_result()
    
    def analyze_pdf_text(self, pdf_path: str) -> List[TextAnalysisResult]:
        """Analyze text extracted from PDF"""
        extracted_texts = self.text_extractor.extract_text_from_pdf(pdf_path)
        results = []
        
        for extracted_text in extracted_texts:
            logger.info(f"Analyzing text from page {extracted_text.page_number}")
            result = self.analyze_text(extracted_text.text_content)
            
            # Add page-specific information
            result.analysis_details["page_number"] = extracted_text.page_number
            result.analysis_details["text_type"] = extracted_text.text_type
            result.analysis_details["font_info"] = extracted_text.font_info[:5]  # First 5 fonts
            result.analysis_details["text_content"] = extracted_text.text_content  # Add actual text content
            
            results.append(result)
        
        return results
    
    def _categorize_content(self, semantic_scores: Dict[str, float], keyword_counts: Dict[str, int]) -> Dict[str, float]:
        """Combine semantic and keyword analysis for content categorization"""
        categories = {}
        
        # Use semantic scores as base
        for category, score in semantic_scores.items():
            categories[category] = score
        
        # Boost categories that have keyword matches
        keyword_boost = 0.2
        for category, count in keyword_counts.items():
            # Map keyword categories to semantic categories
            semantic_category = self._map_keyword_to_semantic_category(category)
            if semantic_category in categories:
                categories[semantic_category] += min(keyword_boost * count, 0.5)
        
        # Normalize to [0, 1]
        max_score = max(categories.values()) if categories else 1.0
        if max_score > 1.0:
            categories = {k: v/max_score for k, v in categories.items()}
        
        return categories
    
    def _map_keyword_to_semantic_category(self, keyword_category: str) -> str:
        """Map keyword categories to semantic categories"""
        mapping = {
            "adult_content": "adult_content",
            "violence": "violence",
            "alcohol_drugs": "alcohol_drugs",
            "gambling": "gambling",
            "dating_romance": "dating_romance",
            "western_holidays": "western_culture",
            "inappropriate_clothing": "adult_content",
            "non_marital_relationships": "dating_romance"
        }
        return mapping.get(keyword_category, "other")
    
    def _calculate_overall_risk(self, keyword_risk: str, keyword_confidence: float,
                               semantic_scores: Dict[str, float], risk_keywords: List[Dict]) -> Tuple[str, float]:
        """Calculate overall risk level combining all factors"""
        risk_factors = []
        
        # Keyword-based risk
        keyword_weight = 0.6
        if keyword_risk == "high":
            risk_factors.append(("high", keyword_confidence * keyword_weight))
        elif keyword_risk == "medium":
            risk_factors.append(("medium", keyword_confidence * keyword_weight))
        
        # Semantic-based risk
        semantic_weight = 0.4
        high_risk_semantic = ["adult_content", "violence", "alcohol_drugs"]
        medium_risk_semantic = ["gambling", "dating_romance"]
        
        for category in high_risk_semantic:
            if category in semantic_scores and semantic_scores[category] > 0.5:
                risk_factors.append(("high", semantic_scores[category] * semantic_weight))
        
        for category in medium_risk_semantic:
            if category in semantic_scores and semantic_scores[category] > 0.4:
                risk_factors.append(("medium", semantic_scores[category] * semantic_weight))
        
        # Determine overall risk
        if not risk_factors:
            return "low", 0.1
        
        # Get highest risk level
        risk_levels = [factor[0] for factor in risk_factors]
        confidences = [factor[1] for factor in risk_factors]
        
        if "high" in risk_levels:
            return "high", max(confidences)
        elif "medium" in risk_levels:
            return "medium", max(confidences)
        else:
            return "low", max(confidences)
    
    def _calculate_text_quality(self, text: str, words: List[str], sentences: List[str]) -> float:
        """Calculate text quality score based on various factors"""
        try:
            # Length factor
            length_score = min(1.0, len(words) / 100.0)
            
            # Vocabulary diversity
            unique_words = len(set(word.lower() for word in words))
            diversity_score = unique_words / len(words) if words else 0
            
            # Average sentence length
            valid_sentences = [s for s in sentences if s.strip()]
            avg_sentence_length = np.mean([len(s.split()) for s in valid_sentences]) if valid_sentences else 0
            sentence_score = min(1.0, avg_sentence_length / 15.0)  # 15 words is good average
            
            # Readability (simple heuristic)
            readability_score = min(1.0, (len(words) - text.count(',') - text.count(';')) / len(words)) if words else 0
            
            # Combine scores
            quality = (length_score * 0.2 + diversity_score * 0.3 + 
                      sentence_score * 0.3 + readability_score * 0.2)
            
            return min(1.0, quality)
            
        except Exception as e:
            logger.error(f"Quality calculation failed: {e}")
            return 0.5
    
    def _get_word_frequency(self, words: List[str], top_k: int = 10) -> List[Tuple[str, int]]:
        """Get most frequent words"""
        # Filter out common stop words and short words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        filtered_words = [word.lower() for word in words if len(word) > 2 and word.lower() not in stop_words]
        
        word_freq = Counter(filtered_words)
        return word_freq.most_common(top_k)
    
    def _get_top_semantic_categories(self, semantic_scores: Dict[str, float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top semantic categories"""
        sorted_categories = sorted(semantic_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_categories[:top_k]
    
    def _analyze_risk_distribution(self, risk_keywords: List[Dict]) -> Dict:
        """Analyze distribution of risk keywords"""
        if not risk_keywords:
            return {"total": 0, "by_severity": {}, "by_category": {}}
        
        severity_dist = Counter([kw["severity"] for kw in risk_keywords])
        category_dist = Counter([kw["category"] for kw in risk_keywords])
        
        return {
            "total": len(risk_keywords),
            "by_severity": dict(severity_dist),
            "by_category": dict(category_dist),
            "unique_keywords": len(set([kw["keyword"] for kw in risk_keywords]))
        }
    
    def _analyze_text_complexity(self, text: str) -> Dict:
        """Analyze text complexity metrics"""
        try:
            words = text.split()
            sentences = re.split(r'[.!?]+', text)
            
            # Average word length
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            
            # Average sentence length
            valid_sentences = [s for s in sentences if s.strip()]
            avg_sentence_length = np.mean([len(s.split()) for s in valid_sentences]) if valid_sentences else 0
            
            # Punctuation density
            punctuation_count = sum(1 for char in text if char in '.,;:!?')
            punctuation_density = punctuation_count / len(text) if text else 0
            
            return {
                "avg_word_length": round(avg_word_length, 2),
                "avg_sentence_length": round(avg_sentence_length, 2),
                "punctuation_density": round(punctuation_density, 3),
                "total_characters": len(text),
                "complexity_score": min(1.0, (avg_word_length * avg_sentence_length) / 100.0)
            }
            
        except Exception as e:
            logger.error(f"Complexity analysis failed: {e}")
            return {}
    
    def _create_empty_result(self) -> TextAnalysisResult:
        """Create empty result for failed analysis"""
        return TextAnalysisResult(
            total_words=0,
            total_sentences=0,
            language="unknown",
            sentiment_score=0.0,
            risk_keywords=[],
            content_categories={},
            semantic_similarities={},
            risk_level="low",
            confidence=0.0,
            extracted_entities=[],
            text_quality_score=0.0,
            analysis_details={}
        )

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python nlp_analyzer.py <pdf_path_or_text>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    analyzer = NLPAnalyzer()
    
    if input_path.endswith('.pdf'):
        # Analyze PDF
        results = analyzer.analyze_pdf_text(input_path)
        print(f"\nAnalyzed {len(results)} pages:")
        for i, result in enumerate(results):
            print(f"\nPage {i+1}:")
            print(f"  Risk Level: {result.risk_level}")
            print(f"  Words: {result.total_words}")
            print(f"  Risk Keywords: {len(result.risk_keywords)}")
            print(f"  Language: {result.language}")
    else:
        # Analyze text directly
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        result = analyzer.analyze_text(text)
        print(f"\nText Analysis Results:")
        print(f"Risk Level: {result.risk_level}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Words: {result.total_words}")
        print(f"Language: {result.language}")
        print(f"Sentiment: {result.sentiment_score:.2f}")
        print(f"Risk Keywords Found: {len(result.risk_keywords)}")
        
        if result.risk_keywords:
            print("\nRisk Keywords:")
            for kw in result.risk_keywords[:10]:  # Show first 10
                print(f"  {kw['keyword']} ({kw['category']}, {kw['severity']})")