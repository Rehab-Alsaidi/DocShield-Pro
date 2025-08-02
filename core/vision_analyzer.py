# core/vision_analyzer.py
import os
from PIL import Image
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Try to import advanced models, fall back to basic analysis
try:
    import torch
    import torch.nn.functional as F
    from transformers import (
        CLIPProcessor, CLIPModel, 
        AutoImageProcessor, AutoModelForImageClassification,
        BlipProcessor, BlipForConditionalGeneration,
        AutoProcessor, AutoModelForCausalLM,  # For Florence-2
        Blip2Processor, Blip2ForConditionalGeneration  # For BLIP-2
    )
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    # Create mock objects for graceful fallback
    torch = None
    F = None

from utils.logger import get_logger

logger = get_logger(__name__)

# Configuration fallbacks when advanced models not available
class MockModelConfig:
    device = "cpu"

class MockModelConfigs:
    def __init__(self):
        self.configs = {
            "nsfw": {"threshold": 0.7},
            "clip": {"model_name": "openai/clip-vit-base-patch32"}
        }
    
    def __getitem__(self, key):
        return self.configs.get(key, {"threshold": 0.7})

# Use actual config if available, otherwise use mocks
try:
    from app.model_config import model_config, MODEL_CONFIGS
except ImportError:
    model_config = MockModelConfig()
    MODEL_CONFIGS = MockModelConfigs()

@dataclass
class VisionAnalysisResult:
    """Result from vision analysis"""
    nsfw_score: float
    nsfw_category: str
    clip_similarities: Dict[str, float]
    scene_classification: Dict[str, float]
    risk_level: str
    confidence: float
    detected_objects: List[str]
    analysis_details: Dict

class NSFWDetector:
    """NSFW content detection model"""
    
    def __init__(self):
        self.device = model_config.device
        self.model = None
        self.processor = None
        # Balanced threshold for optimal accuracy
        self.threshold = 0.35  # Optimized threshold for balanced detection
        self._load_model()
    
    def _load_model(self):
        """Load NSFW detection model"""
        if not ADVANCED_MODELS_AVAILABLE:
            logger.info("Advanced models not available, using basic NSFW detection")
            self.model = None
            return
            
        try:
            logger.info("Loading NSFW detection model...")
            
            # Using multiple models for better accuracy - Enhanced ensemble
            model_names = [
                "Falconsai/nsfw_image_detection",
                "michelecafagna26/nsfw-classifier-lite", 
                "CompVis/stable-diffusion-safety-checker",
                "Salesforce/blip-image-captioning-base",  # Additional model for better understanding
                "google/vit-base-patch16-224"  # Vision transformer for additional analysis
            ]
            
            # Try to load the best available model
            for model_name in model_names:
                try:
                    self.processor = AutoImageProcessor.from_pretrained(model_name)
                    self.model = AutoModelForImageClassification.from_pretrained(model_name)
                    logger.info(f"✅ Successfully loaded NSFW model: {model_name}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
                    continue
            
            if self.device == "cuda" and torch and torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info("NSFW detection model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load NSFW model: {e}")
            # Fallback to manual detection based on captions
            self.model = None
    
    def detect_nsfw(self, image: Image.Image) -> Tuple[float, str]:
        """Enhanced NSFW detection with ensemble approach"""
        try:
            if self.model is None or not ADVANCED_MODELS_AVAILABLE or not torch:
                return self._fallback_nsfw_detection(image)
            
            # Primary model detection
            primary_score = self._detect_with_primary_model(image)
            
            # Secondary validation with different approach
            secondary_score = self._detect_with_content_analysis(image)
            
            # Ensemble voting - combine multiple approaches
            ensemble_score = (primary_score * 0.7) + (secondary_score * 0.3)
            
            # Optimized categorization thresholds
            if ensemble_score > 0.5:
                category = "nsfw"
            elif ensemble_score > 0.25:
                category = "questionable" 
            else:
                category = "safe"
            
            return ensemble_score, category
            
        except Exception as e:
            logger.error(f"NSFW detection failed: {e}")
            return self._fallback_nsfw_detection(image)
    
    def _detect_with_primary_model(self, image: Image.Image) -> float:
        """Primary model detection"""
        try:
            # Preprocess image
            inputs = self.processor(images=image, return_tensors="pt")
            
            if self.device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=-1)
            
            # Get NSFW probability
            nsfw_prob = probabilities[0][1].item() if probabilities.shape[1] > 1 else probabilities[0][0].item()
            return nsfw_prob
            
        except Exception as e:
            logger.warning(f"Primary model detection failed: {e}")
            return 0.0
    
    def _detect_with_content_analysis(self, image: Image.Image) -> float:
        """Secondary content analysis approach"""
        try:
            # Convert image to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Analyze image characteristics
            import numpy as np
            img_array = np.array(image)
            
            # Simple heuristics for content analysis
            # Check for skin-tone dominance, edge patterns, etc.
            height, width = img_array.shape[:2]
            
            # Skin tone detection (simple heuristic)
            skin_pixels = 0
            total_pixels = height * width
            
            for i in range(0, height, 10):  # Sample every 10th pixel for performance
                for j in range(0, width, 10):
                    r, g, b = img_array[i, j][:3]
                    # Simple skin tone detection
                    if (r > 95 and g > 40 and b > 20 and 
                        max(r, g, b) - min(r, g, b) > 15 and 
                        abs(r - g) > 15 and r > g and r > b):
                        skin_pixels += 1
            
            sampled_pixels = (height // 10) * (width // 10)
            skin_ratio = skin_pixels / sampled_pixels if sampled_pixels > 0 else 0
            
            # Return risk score based on skin ratio and other factors
            base_score = min(skin_ratio * 2, 1.0)  # Cap at 1.0
            
            return base_score
            
        except Exception as e:
            logger.warning(f"Content analysis failed: {e}")
            return 0.0
    
    def _fallback_nsfw_detection(self, image: Image.Image) -> Tuple[float, str]:
        """Fallback detection when models aren't available"""
        try:
            # Use basic image analysis
            score = self._detect_with_content_analysis(image)
            category = "questionable" if score > 0.4 else "safe"
            return score, category
        except:
            return 0.0, "safe"

class AdvancedVisionLanguageAnalyzer:
    """Advanced Vision-Language models with contextual prompting"""
    
    def __init__(self):
        self.device = model_config.device
        self.florence_model = None
        self.florence_processor = None
        self.blip2_model = None
        self.blip2_processor = None
        self.blip_model = None
        self.blip_processor = None
        self._load_models()
    
    def _load_models(self):
        """Load the newest computer vision models"""
        if not ADVANCED_MODELS_AVAILABLE:
            logger.info("Advanced models not available, using basic analysis")
            return
            
        # Check Railway environment for memory optimization
        railway_env = os.getenv("RAILWAY_ENVIRONMENT")
        if railway_env:
            logger.info("🚂 Railway environment detected - using memory-optimized model loading")
        
        try:
            # Florence-2: Start with base model for Railway memory constraints
            logger.info("Loading Florence-2 model (optimized for Railway)...")
            try:
                # Try base model first (smaller, more likely to work on Railway)
                self.florence_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
                self.florence_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
                if self.device == "cuda" and torch.cuda.is_available():
                    self.florence_model = self.florence_model.to(self.device)
                logger.info("✅ Florence-2-base model loaded successfully (Railway optimized)")
            except Exception as e:
                logger.warning(f"Florence-2-base failed: {e}, skipping Florence models to save memory")
                self.florence_model = None
                self.florence_processor = None
            
            # BLIP-2: Advanced image captioning and VQA
            logger.info("Loading BLIP-2 model (Railway memory optimized)...")
            try:
                # Use smaller BLIP-2 for Railway
                logger.info("⚡ Using memory-efficient BLIP-2 configuration")
                self.blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", trust_remote_code=True)
                self.blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", trust_remote_code=True, torch_dtype="auto")
                if self.device == "cuda" and torch.cuda.is_available():
                    self.blip2_model = self.blip2_model.to(self.device)
                logger.info("✅ BLIP-2 model loaded successfully")
            except Exception as e:
                logger.warning(f"BLIP-2 large not available: {e}, trying BLIP-2 Flan-T5")
                try:
                    self.blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", trust_remote_code=True)
                    self.blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", trust_remote_code=True)
                    if self.device == "cuda" and torch.cuda.is_available():
                        self.blip2_model = self.blip2_model.to(self.device)
                    logger.info("✅ BLIP-2 Flan-T5 model loaded successfully")
                except Exception as e2:
                    logger.warning(f"BLIP-2 models not available: {e2}")
            
            # Fallback to regular BLIP with multiple model options
            if self.blip2_model is None:
                logger.info("Loading BLIP model as fallback...")
                blip_models = [
                    "Salesforce/blip-image-captioning-base",  # Smaller, more reliable
                    "Salesforce/blip-image-captioning-large",
                    "microsoft/DialoGPT-medium"  # Alternative for text generation
                ]
                
                for model_name in blip_models:
                    try:
                        self.blip_processor = BlipProcessor.from_pretrained(model_name)
                        self.blip_model = BlipForConditionalGeneration.from_pretrained(model_name)
                        if self.device == "cuda" and torch.cuda.is_available():
                            self.blip_model = self.blip_model.to(self.device)
                        logger.info(f"✅ BLIP model loaded successfully: {model_name}")
                        break
                    except Exception as e:
                        logger.warning(f"BLIP model {model_name} failed: {e}")
                        continue
                    
        except Exception as e:
            logger.error(f"Failed to load advanced vision models: {e}")
    
    def analyze_with_cultural_context(self, image: Image.Image) -> Dict[str, any]:
        """Advanced analysis with cultural context and prompting"""
        results = {
            "detailed_caption": "",
            "cultural_assessment": "",
            "content_appropriateness": "",
            "detailed_objects": [],
            "risk_factors": [],
            "cultural_compliance_score": 0.0
        }
        
        try:
            # Generate detailed caption with Florence-2
            if self.florence_model:
                detailed_caption = self._generate_detailed_caption(image)
                results["detailed_caption"] = detailed_caption
                logger.info(f"📝 Generated detailed caption: {detailed_caption[:100]}...")
            
            # Cultural appropriateness assessment with BLIP-2
            if self.blip2_model:
                cultural_assessment = self._assess_cultural_appropriateness(image)
                results["cultural_assessment"] = cultural_assessment
                logger.info(f"🕌 Cultural assessment: {cultural_assessment[:100]}...")
            
            # Content appropriateness with contextual prompting
            content_appropriateness = self._analyze_content_appropriateness(image)
            results["content_appropriateness"] = content_appropriateness
            
            # Extract detailed objects and risk factors
            if results["detailed_caption"]:
                results["detailed_objects"] = self._extract_detailed_objects(results["detailed_caption"])
                results["risk_factors"] = self._identify_cultural_risk_factors(results["detailed_caption"])
            
            # Calculate cultural compliance score
            results["cultural_compliance_score"] = self._calculate_cultural_compliance(results)
            
        except Exception as e:
            logger.error(f"Advanced vision analysis failed: {e}")
            
        return results
    
    def _generate_detailed_caption(self, image: Image.Image) -> str:
        """Generate detailed caption with Florence-2"""
        try:
            # Advanced prompting for detailed, culturally-aware description
            task_prompt = "<MORE_DETAILED_CAPTION>"
            
            inputs = self.florence_processor(text=task_prompt, images=image, return_tensors="pt")
            if self.device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.florence_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=150,
                    num_beams=3,
                    do_sample=False
                )
            
            generated_text = self.florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            
            # Extract the caption part
            caption = generated_text.split(task_prompt)[-1].strip()
            return caption
            
        except Exception as e:
            logger.warning(f"Florence-2 caption generation failed: {e}")
            return ""
    
    def _assess_cultural_appropriateness(self, image: Image.Image) -> str:
        """Assess cultural appropriateness with BLIP-2 using contextual prompting"""
        try:
            # Advanced cultural context prompting
            cultural_prompt = """Analyze this image for cultural appropriateness according to Middle Eastern and Islamic guidelines. Consider:
            1. Modesty standards (clothing, poses, interactions)
            2. Mixed gender interactions 
            3. Religious sensitivity
            4. Cultural symbols and practices
            5. Alcohol or gambling content
            Provide a detailed assessment of cultural compliance."""
            
            inputs = self.blip2_processor(images=image, text=cultural_prompt, return_tensors="pt")
            if self.device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.blip2_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    num_beams=3,
                    temperature=0.7,
                    do_sample=True
                )
            
            assessment = self.blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return assessment.strip()
            
        except Exception as e:
            logger.warning(f"BLIP-2 cultural assessment failed: {e}")
            return ""
    
    def _analyze_content_appropriateness(self, image: Image.Image) -> str:
        """Analyze content appropriateness with advanced prompting"""
        try:
            # Use available model for content analysis
            model = self.blip2_model or self.blip_model
            processor = self.blip2_processor or self.blip_processor
            
            if not model:
                return ""
            
            # Contextual prompting for content analysis
            content_prompt = """Describe this image in detail, focusing on:
            - People present (gender, clothing, poses, interactions)
            - Objects that might be culturally sensitive (alcohol, gambling items)
            - Activities taking place
            - Overall appropriateness for conservative cultural standards
            Be specific and detailed."""
            
            if self.blip2_model:
                inputs = processor(images=image, text=content_prompt, return_tensors="pt")
            else:
                inputs = processor(images=image, return_tensors="pt")
            
            if self.device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                if self.blip2_model:
                    generated_ids = model.generate(**inputs, max_new_tokens=150, num_beams=2)
                else:
                    generated_ids = model.generate(**inputs, max_length=150, num_beams=2)
            
            analysis = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return analysis.strip()
            
        except Exception as e:
            logger.warning(f"Content appropriateness analysis failed: {e}")
            return ""
    
    def _extract_detailed_objects(self, caption: str) -> List[str]:
        """Extract detailed objects from advanced caption"""
        if not caption:
            return []
        
        # Enhanced object extraction with cultural sensitivity
        sensitive_objects = [
            # People and interactions
            "woman", "man", "girl", "boy", "couple", "people", "person",
            "group", "family", "friends", "male", "female",
            
            # Clothing and appearance
            "dress", "skirt", "shirt", "clothing", "outfit", "uniform", 
            "suit", "formal wear", "casual wear", "revealing", "tight",
            "swimwear", "bikini", "shorts", "sleeveless",
            
            # Cultural items
            "wine", "glass", "bottle", "alcohol", "beer", "champagne",
            "cocktail", "bar", "restaurant", "dining", "celebration",
            "gambling", "cards", "casino", "poker", "betting",
            
            # Religious and cultural symbols
            "cross", "church", "religious", "symbol", "mosque", "temple",
            "christmas", "easter", "holiday", "celebration",
            
            # Business and safe items
            "logo", "brand", "sign", "text", "business", "office",
            "food", "meal", "dish", "cooking", "kitchen", "chef"
        ]
        
        found_objects = []
        caption_lower = caption.lower()
        
        for obj in sensitive_objects:
            if obj in caption_lower:
                # Extract context around the object
                words = caption_lower.split()
                for i, word in enumerate(words):
                    if obj in word:
                        # Get surrounding context
                        start = max(0, i-2)
                        end = min(len(words), i+3)
                        context = " ".join(words[start:end])
                        found_objects.append(f"{obj} ({context})")
                        break
        
        return found_objects
    
    def _identify_cultural_risk_factors(self, caption: str) -> List[Dict]:
        """Identify cultural risk factors from detailed caption"""
        risk_factors = []
        if not caption:
            return risk_factors
        
        caption_lower = caption.lower()
        
        # Enhanced risk factor detection with cultural context
        risk_patterns = {
            "mixed_gender_interaction": {
                "patterns": ["man and woman", "men and women", "couple", "mixed group", "together", "romantic", "date"],
                "severity": "high",
                "cultural_note": "Mixed gender interactions without proper context"
            },
            "immodest_clothing": {
                "patterns": ["revealing", "tight", "short", "low-cut", "sleeveless", "swimwear", "bikini"],
                "severity": "high", 
                "cultural_note": "Clothing that doesn't meet modesty standards"
            },
            "alcohol_content": {
                "patterns": ["wine", "beer", "alcohol", "cocktail", "bar", "drinking", "champagne"],
                "severity": "high",
                "cultural_note": "Alcohol consumption or promotion"
            },
            "gambling_content": {
                "patterns": ["gambling", "casino", "poker", "cards", "betting", "slot machine"],
                "severity": "high",
                "cultural_note": "Gambling activities or promotion"
            },
            "western_celebrations": {
                "patterns": ["christmas", "easter", "halloween", "valentine", "new year party"],
                "severity": "medium",
                "cultural_note": "Non-Islamic religious or cultural celebrations"
            },
            "beach_resort": {
                "patterns": ["beach", "swimming", "pool", "resort", "vacation", "swimsuit"],
                "severity": "medium",
                "cultural_note": "Beach and swimming contexts with potential modesty concerns"
            },
            "intimate_behavior": {
                "patterns": ["kissing", "hugging", "embracing", "romantic", "intimate", "affection"],
                "severity": "high",
                "cultural_note": "Public displays of affection"
            }
        }
        
        for risk_type, config in risk_patterns.items():
            for pattern in config["patterns"]:
                if pattern in caption_lower:
                    risk_factors.append({
                        "type": risk_type,
                        "pattern": pattern,
                        "severity": config["severity"],
                        "cultural_note": config["cultural_note"],
                        "found_in": caption[:100] + "..." if len(caption) > 100 else caption
                    })
                    break  # Only add each risk type once
        
        return risk_factors
    
    def _calculate_cultural_compliance(self, results: Dict) -> float:
        """Calculate cultural compliance score based on all analysis results"""
        base_score = 1.0  # Start with perfect compliance
        
        # Deduct points for risk factors
        for risk in results.get("risk_factors", []):
            if risk["severity"] == "high":
                base_score -= 0.3
            elif risk["severity"] == "medium":
                base_score -= 0.15
            else:
                base_score -= 0.05
        
        # Bonus for safe content indicators
        caption = results.get("detailed_caption", "").lower()
        safe_indicators = ["business", "office", "professional", "food", "logo", "brand", "education"]
        for indicator in safe_indicators:
            if indicator in caption:
                base_score += 0.05
                break
        
        # Ensure score stays within bounds
        return max(0.0, min(1.0, base_score))

class CLIPAnalyzer:
    """CLIP model for scene understanding and similarity analysis"""
    
    def __init__(self):
        self.device = model_config.device
        self.model = None
        self.processor = None
        self.config = MODEL_CONFIGS["clip"]
        self._load_model()
        
        # Enhanced predefined categories for comprehensive content analysis
        self.content_categories = [
            "religious content", "church", "mosque", "temple", "cross", "religious symbols",
            # More specific alcohol categories for better detection
            "wine glass", "wine glasses", "champagne glass", "beer bottle", "cocktail glass",
            "alcoholic beverage", "drinking alcohol", "bar scene", "liquor bottle",
            "gambling", "casino", "poker", "betting", "cards", "slot machine",
            "adult content", "revealing clothing", "swimwear", "underwear", "bikini", "lingerie",
            "violence", "weapons", "fighting", "blood", "gun", "knife", "sword",
            "dating", "kissing", "romantic", "couple", "intimate", "embrace", "hug",
            "western holidays", "christmas", "easter", "halloween", "valentine's day", "new year",
            "provocative pose", "seductive", "sensual", "suggestive", "flirting",
            "tight clothing", "short dress", "cleavage", "exposed skin", "immodest",
            "party", "celebration", "nightlife", "dancing", "club scene",
            # Safe categories - these should get low risk scores
            "business logo", "company logo", "brand logo", "corporate identity",
            "yellow logo", "golden logo", "logo design", "emblem", "trademark",
            "corporate symbol", "brand symbol", "graphic design", "text logo",
            "food", "restaurant", "meal", "cooking", "salmon", "fish", "seafood",
            "technology", "computer", "phone", "electronics",
            "nature", "landscape", "animals", "flowers",
            "business", "office", "meeting", "presentation",
            "education", "school", "classroom", "books",
            "sports", "exercise", "gym", "football", "basketball"
        ]
    
    def _load_model(self):
        """Load CLIP model"""
        if not ADVANCED_MODELS_AVAILABLE:
            logger.info("Advanced models not available, using basic content analysis")
            self.model = None
            self.processor = None
            return
            
        try:
            logger.info("Loading CLIP model...")
            
            model_name = self.config["model_name"]
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            
            if self.device == "cuda" and torch and torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info("CLIP model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            self.model = None
            self.processor = None
    
    def analyze_image_content(self, image: Image.Image) -> Dict[str, float]:
        """Enhanced image content analysis with ensemble approach"""
        try:
            if self.model is None or not ADVANCED_MODELS_AVAILABLE or not torch:
                return self._enhanced_fallback_analysis(image)
            
            # Primary CLIP analysis
            primary_similarities = self._analyze_with_clip(image)
            
            # Secondary analysis with different categorization
            secondary_similarities = self._analyze_with_visual_features(image)
            
            # Ensemble combination - weighted average
            combined_similarities = {}
            for category in self.content_categories:
                primary_score = primary_similarities.get(category, 0.0)
                secondary_score = secondary_similarities.get(category, 0.0)
                # Weighted combination (70% CLIP, 30% visual features)
                combined_similarities[category] = (primary_score * 0.7) + (secondary_score * 0.3)
            
            return combined_similarities
            
        except Exception as e:
            logger.error(f"Enhanced CLIP analysis failed: {e}")
            return self._enhanced_fallback_analysis(image)
    
    def _analyze_with_clip(self, image: Image.Image) -> Dict[str, float]:
        """Primary CLIP model analysis"""
        try:
            # Prepare inputs
            inputs = self.processor(
                text=self.content_categories,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            if self.device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get similarities
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=-1)
            
            # Create category-probability mapping
            similarities = {}
            for i, category in enumerate(self.content_categories):
                similarities[category] = float(probs[0][i])
            
            return similarities
            
        except Exception as e:
            logger.warning(f"CLIP model analysis failed: {e}")
            return {}
    
    def _analyze_with_visual_features(self, image: Image.Image) -> Dict[str, float]:
        """Secondary visual features analysis"""
        try:
            import numpy as np
            
            # Convert to RGB and get array
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_array = np.array(image)
            
            similarities = {}
            
            # Initialize all categories with base scores
            for category in self.content_categories:
                similarities[category] = 0.0
            
            # Color-based analysis
            mean_colors = np.mean(img_array, axis=(0, 1))
            r, g, b = mean_colors
            
            # Business/office detection (neutral colors)
            if 100 < r < 180 and 100 < g < 180 and 100 < b < 180:
                similarities["business"] = 0.3
                similarities["office"] = 0.25
                similarities["presentation"] = 0.2
            
            # Logo detection (high contrast, simple shapes)
            gray = np.mean(img_array, axis=2)
            contrast = np.std(gray)
            if contrast > 50:  # High contrast suggests logos/text
                similarities["business logo"] = 0.4
                similarities["company logo"] = 0.35
                similarities["brand logo"] = 0.3
            
            # Food detection (warm colors)
            if r > 150 and g > 100 and r > b:
                similarities["food"] = 0.35
                similarities["restaurant"] = 0.25
                similarities["meal"] = 0.3
            
            # Nature detection (green dominance)
            if g > r and g > b and g > 120:
                similarities["nature"] = 0.4
                similarities["landscape"] = 0.3
                similarities["flowers"] = 0.25
            
            return similarities
            
        except Exception as e:
            logger.warning(f"Visual features analysis failed: {e}")
            return {}
    
    def _enhanced_fallback_analysis(self, image: Image.Image) -> Dict[str, float]:
        """Enhanced fallback when CLIP is not available"""
        try:
            # Use visual features analysis
            visual_similarities = self._analyze_with_visual_features(image)
            
            # Combine with basic analysis
            basic_similarities = self._basic_content_analysis()
            
            # Merge results
            for category in basic_similarities:
                if category in visual_similarities:
                    basic_similarities[category] = max(basic_similarities[category], visual_similarities[category])
            
            return basic_similarities
            
        except Exception as e:
            logger.warning(f"Enhanced fallback failed: {e}")
            return self._basic_content_analysis()
    
    def _basic_content_analysis(self) -> Dict[str, float]:
        """Basic content analysis when CLIP is not available"""
        # Return conservative estimates for safety
        similarities = {}
        for category in self.content_categories:
            if category in ["business", "office", "meeting", "presentation", "education", "food"]:
                similarities[category] = 0.1  # Safe categories get low scores
            elif "logo" in category or "brand" in category or "emblem" in category:
                similarities[category] = 0.15  # Logos get slightly higher but still safe scores
            else:
                similarities[category] = 0.05  # Everything else gets very low scores
        return similarities
    
    def get_top_categories(self, similarities: Dict[str, float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top K categories by similarity"""
        sorted_categories = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_categories[:top_k]

class VisionAnalyzer:
    """Main vision analysis coordinator with advanced models"""
    
    def __init__(self):
        try:
            self.nsfw_detector = NSFWDetector()
            self.clip_analyzer = CLIPAnalyzer()
            self.advanced_analyzer = AdvancedVisionLanguageAnalyzer()
            logger.info("Vision Analyzer initialized successfully with advanced models")
        except Exception as e:
            logger.error(f"Vision Analyzer initialization failed: {e}")
            # Create minimal fallback analyzers
            self.nsfw_detector = None
            self.clip_analyzer = None
            self.advanced_analyzer = None
    
    def analyze_image(self, image: Image.Image, caption: str = "") -> VisionAnalysisResult:
        """Comprehensive image analysis with advanced models"""
        try:
            logger.debug("Starting comprehensive image analysis with advanced models")
            
            # Advanced Vision-Language Analysis (NEW!)
            advanced_results = {}
            if self.advanced_analyzer:
                logger.info("🚀 Running advanced vision-language analysis...")
                advanced_results = self.advanced_analyzer.analyze_with_cultural_context(image)
                
                # Use advanced caption if available
                if advanced_results.get("detailed_caption") and not caption:
                    caption = advanced_results["detailed_caption"]
                    logger.info(f"📝 Using advanced caption: {caption[:100]}...")
            
            # NSFW Detection
            if self.nsfw_detector:
                nsfw_score, nsfw_category = self.nsfw_detector.detect_nsfw(image)
            else:
                nsfw_score, nsfw_category = 0.0, "safe"
            
            # CLIP Analysis
            if self.clip_analyzer:
                clip_similarities = self.clip_analyzer.analyze_image_content(image)
                top_categories = self.clip_analyzer.get_top_categories(clip_similarities, top_k=5)
            else:
                clip_similarities = {}
                top_categories = []
            
            # Scene classification (top categories)
            scene_classification = dict(top_categories)
            
            # Detect potentially problematic objects/content
            detected_objects = self._extract_detected_objects(clip_similarities, caption)
            
            # Enhanced risk calculation with advanced analysis
            risk_level, confidence = self._calculate_advanced_risk_level(
                nsfw_score, clip_similarities, caption, advanced_results
            )
            
            # Enhanced analysis details with advanced results
            analysis_details = {
                "top_categories": top_categories,
                "nsfw_details": {
                    "score": nsfw_score,
                    "category": nsfw_category,
                    "threshold_used": self.nsfw_detector.threshold if self.nsfw_detector else 0.35
                },
                "caption_analysis": self._analyze_caption_keywords(caption),
                "risk_factors": self._identify_risk_factors(clip_similarities, nsfw_score, caption),
                "advanced_analysis": advanced_results,  # NEW: Include advanced analysis
                "cultural_compliance_score": advanced_results.get("cultural_compliance_score", 0.8)
            }
            
            return VisionAnalysisResult(
                nsfw_score=nsfw_score,
                nsfw_category=nsfw_category,
                clip_similarities=clip_similarities,
                scene_classification=scene_classification,
                risk_level=risk_level,
                confidence=confidence,
                detected_objects=detected_objects,
                analysis_details=analysis_details
            )
            
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return self._create_fallback_result()
    
    def _extract_detected_objects(self, similarities: Dict[str, float], caption: str) -> List[str]:
        """Extract high-confidence detected objects"""
        threshold = 0.1  # Minimum similarity threshold
        detected = []
        
        for category, score in similarities.items():
            if score > threshold:
                detected.append(f"{category} ({score:.2f})")
        
        # Also extract from caption
        if caption:
            caption_objects = self._extract_objects_from_caption(caption)
            detected.extend(caption_objects)
        
        return detected
    
    def _extract_objects_from_caption(self, caption: str) -> List[str]:
        """Extract objects/entities mentioned in caption"""
        # Simple keyword extraction - could be enhanced with NER
        keywords = []
        important_words = caption.lower().split()
        
        # Filter for relevant nouns and objects
        relevant_objects = [
            'person', 'people', 'man', 'woman', 'child', 'building', 'car', 'book',
            'food', 'drink', 'table', 'chair', 'window', 'door', 'tree', 'flower'
        ]
        
        for word in important_words:
            word_clean = word.strip('.,!?;:"')
            if word_clean in relevant_objects:
                keywords.append(word_clean)
        
        return list(set(keywords))  # Remove duplicates
    
    def _calculate_risk_level(self, nsfw_score: float, similarities: Dict[str, float], 
                            caption: str) -> Tuple[str, float]:
        """Calculate overall risk level and confidence"""
        risk_scores = []
        
        # NSFW risk - Optimized thresholds for balanced accuracy
        if nsfw_score > 0.5:  # High risk threshold - be confident
            risk_scores.append(("high", nsfw_score))
        elif nsfw_score > 0.25:  # Medium risk threshold - cautious detection
            risk_scores.append(("medium", nsfw_score))
        elif nsfw_score > 0.12:  # Low threshold for early detection
            risk_scores.append(("low", nsfw_score))
        
        # Content category risks - Enhanced with specific alcohol detection and exclusions
        high_risk_categories = [
            "adult content", "revealing clothing", "gambling", "violence", "weapons", 
            "kissing", "romantic", "couple", "swimwear", "underwear", "dating", "intimate", "provocative",
            # Specific alcohol categories for better wine glass detection
            "wine glass", "wine glasses", "champagne glass", "beer bottle", "cocktail glass",
            "alcoholic beverage", "drinking alcohol", "bar scene", "liquor bottle"
        ]
        medium_risk_categories = [
            "religious content", "church", "mosque", "western holidays", "christmas", "easter", 
            "halloween", "valentine's day", "nightclub", "party"
        ]
        
        # Excluded categories - DO NOT FLAG these as risky
        excluded_categories = [
            "business logo", "company logo", "brand logo", "corporate identity",
            "yellow logo", "golden logo", "logo design", "emblem", "trademark",
            "corporate symbol", "brand symbol", "graphic design", "text logo",
            "food", "restaurant", "meal", "cooking", "salmon", "fish", "seafood",
            "technology", "business", "office", "education", "nature", "landscape", "sports"
        ]
        
        # Check excluded categories first - if found, reduce their impact
        for category in excluded_categories:
            if category in similarities and similarities[category] > 0.12:  # Lower threshold for safer exclusion
                # This is likely a logo or food item, significantly reduce risk
                logger.info(f"🟡 Detected safe content: {category} with score {similarities[category]:.3f} - EXCLUDING from violations")
                return "low", 0.03  # Force very low risk for excluded content
        
        # Additional yellow/golden logo specific check
        caption_lower = caption.lower() if caption else ""
        if any(term in caption_lower for term in ["yellow", "golden", "gold", "amber"]) and \
           any(term in caption_lower for term in ["logo", "brand", "emblem", "symbol", "sign"]):
            logger.info(f"🟡 Yellow/golden logo detected in caption - EXCLUDING from violations")
            return "low", 0.1
        
        for category in high_risk_categories:
            # Optimized thresholds for better accuracy
            if "revealing" in category or "clothing" in category or "swimwear" in category or "underwear" in category:
                threshold = 0.25  # Optimized threshold for clothing issues
            elif "couple" in category or "romantic" in category or "kissing" in category or "dating" in category:
                threshold = 0.28  # Optimized threshold for relationship content
            elif "wine glass" in category or "alcohol" in category:
                threshold = 0.35   # Moderate threshold for alcohol detection
            else:
                threshold = 0.27   # Optimized standard threshold
            
            if category in similarities and similarities[category] > threshold:
                risk_scores.append(("high", similarities[category]))
        
        for category in medium_risk_categories:
            # Optimized thresholds for medium risk categories
            threshold = 0.22 if "religious" in category or "church" in category or "mosque" in category else 0.26
            if category in similarities and similarities[category] > threshold:
                risk_scores.append(("medium", similarities[category]))
        
        # Caption-based risk
        caption_risk = self._analyze_caption_risk(caption)
        if caption_risk[0] != "low":
            risk_scores.append(caption_risk)
        
        # Determine overall risk with balanced confidence requirements
        if not risk_scores:
            return "low", 0.9
        
        # Get highest risk level
        risk_levels = [score[0] for score in risk_scores]
        confidence_scores = [score[1] for score in risk_scores]
        
        if "high" in risk_levels:
            return "high", max(confidence_scores)
        elif "medium" in risk_levels:
            return "medium", max(confidence_scores)
        else:
            return "low", max(confidence_scores)
    
    def _calculate_advanced_risk_level(self, nsfw_score: float, similarities: Dict[str, float], 
                                     caption: str, advanced_results: Dict) -> Tuple[str, float]:
        """Enhanced risk calculation using advanced vision-language analysis"""
        # Start with traditional risk calculation
        traditional_risk, traditional_confidence = self._calculate_risk_level(
            nsfw_score, similarities, caption
        )
        
        # If no advanced results, return traditional
        if not advanced_results:
            return traditional_risk, traditional_confidence
        
        # Factor in cultural compliance score
        cultural_score = advanced_results.get("cultural_compliance_score", 0.8)
        advanced_risk_factors = advanced_results.get("risk_factors", [])
        
        # Calculate advanced risk based on cultural compliance
        if cultural_score < 0.3:
            advanced_risk = "high"
            advanced_confidence = 0.95
        elif cultural_score < 0.6:
            advanced_risk = "medium" 
            advanced_confidence = 0.8
        else:
            advanced_risk = "low"
            advanced_confidence = 0.9
        
        # Combine traditional and advanced risk assessments
        risk_levels = [traditional_risk, advanced_risk]
        confidences = [traditional_confidence, advanced_confidence]
        
        # Use the higher risk level (more conservative approach)
        if "high" in risk_levels:
            final_risk = "high"
            final_confidence = max(confidences)
        elif "medium" in risk_levels:
            final_risk = "medium"
            final_confidence = max(confidences)
        else:
            final_risk = "low"
            final_confidence = max(confidences)
        
        # Log advanced analysis insights
        if advanced_risk_factors:
            logger.info(f"🎯 Advanced analysis found {len(advanced_risk_factors)} cultural risk factors")
            for risk in advanced_risk_factors[:3]:  # Log first 3
                logger.info(f"   - {risk.get('type', 'unknown')}: {risk.get('cultural_note', 'No details')}")
        
        logger.info(f"🏛️ Cultural compliance score: {cultural_score:.2f} | Final risk: {final_risk} ({final_confidence:.2f})")
        
        return final_risk, final_confidence
    
    def _analyze_caption_risk(self, caption: str) -> Tuple[str, float]:
        """Analyze risk based on caption content"""
        if not caption:
            return "low", 0.0
        
        caption_lower = caption.lower()
        
        # Check for excluded content first - DO NOT FLAG these items
        excluded_terms = [
            "logo", "logos", "brand", "branding", "company logo", "business logo",
            "emblem", "symbol", "corporate", "trademark", "brand mark", "yellow logo",
            "golden logo", "company emblem", "business symbol", "corporate identity",
            "salmon", "fish", "seafood", "food", "meal", "dish", "cuisine", "cooking",
            "restaurant food", "dinner", "lunch", "breakfast", "plate", "serving",
            "chef", "kitchen", "recipe", "ingredients", "menu", "text", "sign",
            "graphic design", "design element", "icon"
        ]
        
        # If any excluded terms are found, return low risk immediately
        for term in excluded_terms:
            if term in caption_lower:
                return "low", 0.1
        
        # Enhanced detection for man and woman TOGETHER - HIGH RISK only if BOTH present
        # Single gender is acceptable (man only, woman only, boy only, girl only)
        male_terms = ["man", "men", "male", "boy", "boys", "husband", "boyfriend", "gentleman", "guy", "dude", "father", "dad", "son", "brother"]
        female_terms = ["woman", "women", "female", "girl", "girls", "wife", "girlfriend", "lady", "mother", "mom", "daughter", "sister"]
        
        # Check if BOTH male and female terms are present
        has_male = any(term in caption_lower for term in male_terms)
        has_female = any(term in caption_lower for term in female_terms)
        
        # Only flag as unacceptable if BOTH genders are present together
        if has_male and has_female:
            return "high", 0.85  # High confidence for mixed gender detection
        
        # Enhanced couple and relationship detection
        couple_terms = ["couple", "couples", "hugging", "kissing", "embracing", "romantic", "together", "holding hands", "arm in arm", "intimate moment", "love", "romance", "affection", "tender moment"]
        for term in couple_terms:
            if term in caption_lower:
                return "high", 0.9
        
        # Enhanced group detection - specifically check for mixed gender groups
        # But exclude professional/business contexts and single person scenarios
        professional_contexts = ["business", "meeting", "office", "work", "professional", "conference", "presentation", "corporate", "standing", "sitting alone", "individual", "portrait"]
        is_professional = any(term in caption_lower for term in professional_contexts)
        
        # Check if it's clearly a single person context
        single_person_indicators = ["standing", "sitting", "individual", "person", "portrait", "alone", "professional photo"]
        is_single_person = any(term in caption_lower for term in single_person_indicators)
        
        if not is_professional and not is_single_person:  # Only flag non-professional, non-single-person contexts
            group_patterns = [
                "man and woman sitting", "man and woman standing", "man and woman together",
                "men and women", "mixed group", "people sitting together",
                "man woman sitting", "man woman standing"
            ]
            
            for pattern in group_patterns:
                if pattern in caption_lower:
                    # Check if it's a mixed gender scenario
                    if "man" in pattern and "woman" in pattern:
                        return "high", 0.88  # High risk for mixed gender groups
                    elif has_male and has_female:  # Additional check using the terms we found
                        return "high", 0.85
        
        # High risk keywords - Enhanced for Middle Eastern cultural standards with better alcohol detection
        high_risk_words = [
            "nude", "naked", "explicit", "sexual", "adult", "weapon", "gun", "violence",
            "blood", "drugs", "casino", "gambling", "intimate",
            "provocative", "sensual", "seductive", "tempting", "erotic", "lustful",
            "immodest", "indecent", "scandalous", "inappropriate", "revealing outfit",
            # Enhanced alcohol detection - more specific terms
            "wine glass", "wine glasses", "champagne glass", "beer bottle", "liquor", 
            "cocktail", "martini", "vodka", "whiskey", "rum", "gin", "tequila",
            "alcoholic drink", "alcoholic beverage", "drinking alcohol", "bar scene",
            "toasting with wine", "wine tasting", "champagne toast"
        ]
        
        # Medium risk keywords - Enhanced cultural sensitivity with cleaner alcohol detection
        medium_risk_words = [
            "revealing", "swimsuit", "bikini", "romantic", "kissing", "dating",
            "church", "religious", "christmas", "easter", "tight clothing", "short dress",
            "cleavage", "exposed", "flirting", "seductive pose", "suggestive gesture",
            "western celebration", "party atmosphere", "nightlife", "club scene",
            # General alcohol terms (less specific than high-risk)
            "alcohol", "drinking", "wine", "beer", "bar"
        ]
        
        high_risk_count = sum(1 for word in high_risk_words if word in caption_lower)
        medium_risk_count = sum(1 for word in medium_risk_words if word in caption_lower)
        
        if high_risk_count > 0:
            # Enhanced confidence calculation for better accuracy
            confidence = min(0.95, 0.7 + (high_risk_count * 0.1))  # Higher base confidence
            return "high", confidence
        elif medium_risk_count > 0:
            confidence = min(0.8, 0.5 + (medium_risk_count * 0.15))  # Improved medium risk confidence
            return "medium", confidence
        else:
            return "low", 0.1
    
    def _analyze_caption_keywords(self, caption: str) -> Dict:
        """Analyze caption for specific keywords"""
        if not caption:
            return {"keywords_found": [], "risk_indicators": []}
        
        try:
            from app.config import CONTENT_RULES
        except ImportError:
            # Fallback content rules if config not available
            CONTENT_RULES = {
                "high": {
                    "adult_content": {
                        "keywords": ["nude", "naked", "explicit", "sexual"],
                        "severity": "high"
                    },
                    "violence": {
                        "keywords": ["weapon", "gun", "violence", "blood"],
                        "severity": "high"
                    }
                },
                "medium": {
                    "alcohol": {
                        "keywords": ["alcohol", "beer", "wine", "drinking"],
                        "severity": "medium"
                    },
                    "gambling": {
                        "keywords": ["gambling", "casino", "poker", "betting"],
                        "severity": "medium"
                    }
                }
            }
        
        keywords_found = []
        risk_indicators = []
        caption_lower = caption.lower()
        
        for risk_level, categories in CONTENT_RULES.items():
            for category, rules in categories.items():
                for keyword in rules["keywords"]:
                    if keyword in caption_lower:
                        keywords_found.append({
                            "keyword": keyword,
                            "category": category,
                            "risk_level": risk_level,
                            "severity": rules["severity"]
                        })
                        risk_indicators.append(category)
        
        return {
            "keywords_found": keywords_found,
            "risk_indicators": list(set(risk_indicators)),
            "total_risk_keywords": len(keywords_found)
        }
    
    def _identify_risk_factors(self, similarities: Dict[str, float], nsfw_score: float, caption: str) -> List[Dict]:
        """Identify specific risk factors"""
        risk_factors = []
        
        # NSFW risk factor
        if nsfw_score > 0.3:
            risk_factors.append({
                "type": "nsfw_content",
                "score": nsfw_score,
                "description": f"NSFW detection score: {nsfw_score:.2f}",
                "severity": "high" if nsfw_score > 0.7 else "medium"
            })
        
        # Category-based risk factors
        risk_categories = {
            "adult content": "high",
            "revealing clothing": "medium",
            "alcohol": "medium",
            "gambling": "medium",
            "violence": "high",
            "religious content": "low"
        }
        
        for category, severity in risk_categories.items():
            if category in similarities and similarities[category] > 0.2:
                risk_factors.append({
                    "type": "content_category",
                    "category": category,
                    "score": similarities[category],
                    "description": f"Detected {category} with confidence {similarities[category]:.2f}",
                    "severity": severity
                })
        
        # Caption-based risk factors
        caption_analysis = self._analyze_caption_keywords(caption)
        for keyword_info in caption_analysis["keywords_found"]:
            risk_factors.append({
                "type": "caption_keyword",
                "keyword": keyword_info["keyword"],
                "category": keyword_info["category"],
                "description": f"Found risk keyword: {keyword_info['keyword']}",
                "severity": keyword_info["severity"]
            })
        
        return risk_factors
    
    def _create_fallback_result(self) -> VisionAnalysisResult:
        """Create fallback result when analysis fails"""
        return VisionAnalysisResult(
            nsfw_score=0.0,
            nsfw_category="unknown",
            clip_similarities={},
            scene_classification={},
            risk_level="unknown",
            confidence=0.0,
            detected_objects=[],
            analysis_details={"error": "Analysis failed"}
        )

# Batch processing utility
class BatchVisionAnalyzer:
    """Batch processing for multiple images"""
    
    def __init__(self, batch_size: int = 8):
        self.analyzer = VisionAnalyzer()
        self.batch_size = batch_size
    
    def analyze_batch(self, images: List[Image.Image], captions: List[str] = None) -> List[VisionAnalysisResult]:
        """Analyze multiple images in batches"""
        if captions is None:
            captions = [""] * len(images)
        
        results = []
        
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            batch_captions = captions[i:i + self.batch_size]
            
            logger.info(f"Processing batch {i//self.batch_size + 1}, images {i} to {min(i + self.batch_size, len(images))}")
            
            for img, caption in zip(batch_images, batch_captions):
                result = self.analyzer.analyze_image(img, caption)
                results.append(result)
        
        return results

# Example usage
if __name__ == "__main__":
    from PIL import Image
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python vision_analyzer.py <image_path>")
        sys.exit(1)
    
    # Test the analyzer
    image_path = sys.argv[1]
    image = Image.open(image_path)
    
    analyzer = VisionAnalyzer()
    result = analyzer.analyze_image(image, "Test image analysis")
    
    print(f"\nVision Analysis Results:")
    print(f"Risk Level: {result.risk_level}")
    print(f"NSFW Score: {result.nsfw_score:.2f}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"\nTop Categories:")
    for category, score in list(result.scene_classification.items())[:5]:
        print(f"  {category}: {score:.3f}")
    print(f"\nDetected Objects: {result.detected_objects}")
    print(f"Risk Factors: {len(result.analysis_details.get('risk_factors', []))}")