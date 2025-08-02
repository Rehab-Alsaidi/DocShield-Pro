# app/config.py
import os
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ModelConfig:
    """Configuration for AI models"""
    florence_model: str = "microsoft/Florence-2-base-ft"  # Use fine-tuned version without flash_attn dependency
    clip_model: str = "openai/clip-vit-base-patch32"
    nsfw_model: str = "Falconsai/nsfw_image_detection"
    sentence_transformer: str = "all-MiniLM-L6-v2"
    device: str = "cuda" if os.getenv("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///content_moderator.db")
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30

@dataclass
class AppConfig:
    """Main application configuration"""
    # Flask settings
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "5000"))
    
    # File upload settings - Enhanced for professional use
    max_file_size: int = 10 * 1024 * 1024 * 1024  # 10GB for enterprise PDFs
    allowed_extensions: List[str] = None
    upload_folder: str = "static/uploads"
    chunk_upload_folder: str = "static/chunks"  # For chunked uploads
    temp_processing_folder: str = "static/temp"  # For processing
    
    # Large file processing settings
    chunk_size: int = 16 * 1024 * 1024  # 16MB chunks for streaming
    max_pages_per_batch: int = 10  # Process pages in batches
    enable_progress_tracking: bool = True
    
    # Content moderation settings
    confidence_threshold: float = 0.7
    batch_size: int = 8
    max_image_size: tuple = (1024, 1024)
    
    # Analysis settings
    enable_text_analysis: bool = True
    enable_image_analysis: bool = True
    enable_metadata_analysis: bool = True
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = ['pdf']

# Enhanced content moderation rules for Middle Eastern cultural compliance
CONTENT_RULES = {
    "high_risk": {
        "adult_content": {
            "keywords": ["nude", "naked", "explicit", "sexual", "intimate", "adult content", "provocative", "seductive"],
            "confidence_threshold": 0.7,  # Lowered for stricter detection
            "severity": "high"
        },
        "revealing_clothing": {
            "keywords": ["revealing", "bikini", "underwear", "lingerie", "swimsuit", "short dress", "low cut", "cleavage", "exposed", "skimpy"],
            "confidence_threshold": 0.6,  # Elevated to high risk
            "severity": "high"
        },
        "alcohol_substances": {
            "keywords": ["alcohol", "beer", "wine", "drinking", "drunk", "drugs", "smoking", "cocktail", "bar", "pub", "nightclub"],
            "confidence_threshold": 0.5,  # Lowered threshold, elevated severity
            "severity": "high"
        },
        "gambling": {
            "keywords": ["casino", "gambling", "poker", "betting", "cards", "dice", "slot machine", "lottery", "roulette"],
            "confidence_threshold": 0.6,  # Elevated to high risk
            "severity": "high"
        },
        "violence": {
            "keywords": ["violence", "weapon", "gun", "knife", "blood", "fighting", "war", "assault", "murder", "killing"],
            "confidence_threshold": 0.7,
            "severity": "high"
        }
    },
    "medium_risk": {
        "inappropriate_relationships": {
            "keywords": ["dating", "kissing", "couple", "romantic", "intimate relationship", "boyfriend", "girlfriend", "affair"],
            "confidence_threshold": 0.5,
            "severity": "medium"
        },
        "western_cultural_content": {
            "keywords": ["christmas", "easter", "halloween", "valentine", "santa", "easter bunny", "new year party", "prom"],
            "confidence_threshold": 0.6,
            "severity": "medium"
        },
        "mixed_gender_activities": {
            "keywords": ["mixed dancing", "co-ed party", "beach party", "nightlife", "club scene"],
            "confidence_threshold": 0.6,
            "severity": "medium"
        },
        "immodest_behavior": {
            "keywords": ["dancing provocatively", "twerking", "grinding", "inappropriate touching", "suggestive poses"],
            "confidence_threshold": 0.6,
            "severity": "medium"
        }
    },
    "contextual": {
        "cultural_sensitivity": {
            "keywords": ["wedding", "marriage", "family gathering", "celebration", "traditional dress", "cultural event"],
            "confidence_threshold": 0.4,
            "severity": "low"
        },
        "business_professional": {
            "keywords": ["business meeting", "professional", "office", "conference", "presentation", "workplace"],
            "confidence_threshold": 0.3,
            "severity": "low"
        }
    }
}

# Model configurations
MODEL_CONFIGS = {
    "florence": {
        "model_name": "microsoft/Florence-2-base",
        "task_prompt": "<MORE_DETAILED_CAPTION>",
        "max_new_tokens": 1024,
        "num_beams": 3,
        "torch_dtype": "float16"
    },
    "clip": {
        "model_name": "openai/clip-vit-base-patch32",
        "image_size": 224,
        "batch_size": 16
    },
    "nsfw": {
        "threshold": 0.5,
        "categories": ["safe", "nsfw"]
    }
}

# Initialize configurations
app_config = AppConfig()
model_config = ModelConfig()
db_config = DatabaseConfig()