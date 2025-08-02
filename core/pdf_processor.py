# core/pdf_processor.py
import fitz  # PyMuPDF
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import io
import base64
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import hashlib
import os

from utils.logger import get_logger

# Import config with fallback
try:
    from app.config import model_config, MODEL_CONFIGS
except ImportError:
    # Create fallback config
    class MockModelConfig:
        device = "cpu"
    
    class MockModelConfigs:
        def __getitem__(self, key):
            return {"device": "cpu"}
    
    model_config = MockModelConfig()
    MODEL_CONFIGS = MockModelConfigs()

logger = get_logger(__name__)

@dataclass
class ExtractedImage:
    """Data structure for extracted images"""
    page_number: int
    image_index: int
    image: Image.Image
    size: Tuple[int, int]
    format: str
    image_hash: str
    bbox: Optional[Tuple[float, float, float, float]] = None

@dataclass
class ProcessedImage:
    """Data structure for processed images with analysis"""
    page_number: int
    image_index: int
    size: Tuple[int, int]
    format: str
    caption: str
    confidence: float
    image_base64: str
    image_hash: str
    analysis_metadata: Dict
    violations: List = None  # Will be populated during moderation
    
    def __post_init__(self):
        if self.violations is None:
            self.violations = []

class VisionCaptionModel:
    """Enhanced vision model for image captioning using BLIP"""
    
    def __init__(self):
        self.device = model_config.device if hasattr(model_config, 'device') else "cpu"
        self.model = None
        self.processor = None
        self.model_type = None
        self._load_model()
    
    def _load_model(self):
        """Load the best available vision model for captioning"""
        try:
            logger.info("🤖 Loading vision captioning model...")
            
            # Try BLIP first (most reliable)
            if self._try_load_blip():
                return
            
            # Fallback to BLIP-2 if available  
            if self._try_load_blip2():
                return
                
            # Final fallback to basic analysis
            logger.info("🔄 Using basic image analysis as fallback...")
            self.model = None
            self.processor = None
            self.model_type = "basic"
                    
        except Exception as e:
            logger.error(f"❌ Model initialization error: {e}")
            self.model = None
            self.processor = None
            self.model_type = "basic"
    
    def _try_load_blip(self) -> bool:
        """Try to load BLIP model"""
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            model_name = "Salesforce/blip-image-captioning-base"
            logger.info(f"📥 Loading BLIP model: {model_name}")
            
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)
            
            # Move to appropriate device
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to(self.device)
            else:
                self.model = self.model.to("cpu")
                self.device = "cpu"
            
            self.model.eval()
            self.model_type = "blip"
            
            logger.info("✅ BLIP model loaded successfully")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ BLIP loading failed: {e}")
            return False
    
    def _try_load_blip2(self) -> bool:
        """Try to load BLIP-2 model"""
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            
            model_name = "Salesforce/blip2-opt-2.7b"
            logger.info(f"📥 Loading BLIP-2 model: {model_name}")
            
            self.processor = Blip2Processor.from_pretrained(model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(model_name)
            
            # Move to appropriate device
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to(self.device)
            else:
                self.model = self.model.to("cpu")
                self.device = "cpu"
            
            self.model.eval()
            self.model_type = "blip2"
            
            logger.info("✅ BLIP-2 model loaded successfully")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ BLIP-2 loading failed: {e}")
            return False
    
    def generate_caption(self, image: Image.Image) -> Tuple[str, float]:
        """Generate caption using available vision models"""
        try:
            # Ensure image is in RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Use appropriate model based on what's loaded
            if self.model_type == "blip":
                return self._generate_blip_caption(image)
            elif self.model_type == "blip2":
                return self._generate_blip2_caption(image)
            else:
                return self._generate_basic_caption(image)
            
        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            # Fallback to basic description
            return self._generate_basic_caption(image)
    
    def _generate_blip_caption(self, image: Image.Image) -> Tuple[str, float]:
        """Generate caption using BLIP model"""
        try:
            logger.debug("🎯 Generating caption with BLIP")
            
            # Prepare inputs for BLIP
            inputs = self.processor(image, return_tensors="pt")
            
            # Move inputs to device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            logger.debug("🔄 Running BLIP inference...")
            
            # Generate caption with BLIP
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_length=50, num_beams=5)
            
            # Decode the generated text
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
            # Clean up the caption
            caption = caption.strip()
            if not caption:
                caption = "Image content detected"
            
            # Simple confidence calculation based on caption quality
            confidence = min(0.95, max(0.4, len(caption.split()) / 15.0))
            
            logger.debug(f"✅ BLIP caption generated: {caption}")
            
            return caption, confidence
            
        except Exception as e:
            logger.error(f"❌ BLIP caption generation failed: {e}")
            return self._generate_basic_caption(image)
    
    def _generate_blip2_caption(self, image: Image.Image) -> Tuple[str, float]:
        """Generate caption using BLIP-2 model"""
        try:
            logger.debug("🎯 Generating caption with BLIP-2")
            
            # Prepare inputs for BLIP-2
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Move inputs to device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            logger.debug("🔄 Running BLIP-2 inference...")
            
            # Generate caption with BLIP-2
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_length=50, num_beams=5)
            
            # Decode the generated text
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
            # Clean up the caption
            caption = caption.strip()
            if not caption:
                caption = "Image content detected"
            
            # Simple confidence calculation based on caption quality
            confidence = min(0.95, max(0.4, len(caption.split()) / 15.0))
            
            logger.debug(f"✅ BLIP-2 caption generated: {caption}")
            
            return caption, confidence
            
        except Exception as e:
            logger.error(f"❌ BLIP-2 caption generation failed: {e}")
            return self._generate_basic_caption(image)
    
    def _generate_basic_caption(self, image: Image.Image) -> Tuple[str, float]:
        """Generate basic caption using simple image analysis"""
        try:
            logger.info("Using basic image analysis")
            
            # Basic image analysis
            width, height = image.size
            aspect_ratio = width / height
            
            # Simple description based on image properties
            if aspect_ratio > 1.5:
                description = "Wide format document content"
            elif aspect_ratio < 0.7:
                description = "Tall format document content"
            else:
                description = "Standard format document content"
            
            # Add size information
            if width > 1000 or height > 1000:
                description += " - high resolution"
            elif width < 300 or height < 300:
                description += " - low resolution"
            else:
                description += " - medium resolution"
            
            # Basic confidence score
            confidence = 0.6
            
            logger.info(f"✅ CLIP-based description: '{description}' (confidence: {confidence:.2f})")
            return description, confidence
            
        except Exception as e:
            logger.error(f"❌ CLIP caption generation failed: {e}")
            return "Document image content detected", 0.5

class PDFProcessor:
    """Main PDF processing class"""
    
    def __init__(self):
        self.vision_model = VisionCaptionModel()
        logger.info("PDF Processor initialized")
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[ExtractedImage]:
        """Extract all images from PDF with metadata"""
        extracted_images = []
        
        try:
            pdf_document = fitz.open(pdf_path)
            logger.info(f"Processing PDF with {pdf_document.page_count} pages")
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                image_list = page.get_images()
                
                logger.debug(f"Found {len(image_list)} images on page {page_num + 1}")
                
                for img_index, img in enumerate(image_list):
                    try:
                        extracted_image = self._extract_single_image(
                            pdf_document, img, page_num, img_index
                        )
                        if extracted_image:
                            extracted_images.append(extracted_image)
                            
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")
                        continue
            
            pdf_document.close()
            logger.info(f"Successfully extracted {len(extracted_images)} images from PDF")
            return extracted_images
            
        except Exception as e:
            logger.error(f"Failed to process PDF: {e}")
            return []
    
    def _extract_single_image(self, pdf_doc, img_info, page_num: int, img_index: int) -> Optional[ExtractedImage]:
        """Extract a single image from PDF"""
        try:
            xref = img_info[0]
            base_image = pdf_doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Skip very small images (likely decorative elements)
            min_size = 50
            if image.size[0] < min_size or image.size[1] < min_size:
                logger.debug(f"Skipping small image: {image.size}")
                return None
            
            # Generate image hash for deduplication
            image_hash = hashlib.md5(image_bytes).hexdigest()
            
            return ExtractedImage(
                page_number=page_num + 1,
                image_index=img_index,
                image=image,
                size=image.size,
                format=base_image.get("ext", "unknown"),
                image_hash=image_hash
            )
            
        except Exception as e:
            logger.error(f"Failed to extract image: {e}")
            return None
    
    def process_images(self, extracted_images: List[ExtractedImage]) -> List[ProcessedImage]:
        """Process extracted images with Florence-2"""
        processed_images = []
        
        for extracted_img in extracted_images:
            try:
                logger.info(f"Processing image {extracted_img.image_index} from page {extracted_img.page_number}")
                
                # Generate caption
                caption, confidence = self.vision_model.generate_caption(extracted_img.image)
                
                # Convert image to base64 for web display
                image_base64 = self._image_to_base64(extracted_img.image)
                
                # Create analysis metadata
                analysis_metadata = {
                    "processing_timestamp": str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu",
                    "model_used": "Florence-2-base",
                    "image_dimensions": extracted_img.size,
                    "file_format": extracted_img.format
                }
                
                processed_image = ProcessedImage(
                    page_number=extracted_img.page_number,
                    image_index=extracted_img.image_index,
                    size=extracted_img.size,
                    format=extracted_img.format,
                    caption=caption,
                    confidence=confidence,
                    image_base64=image_base64,
                    image_hash=extracted_img.image_hash,
                    analysis_metadata=analysis_metadata
                )
                
                processed_images.append(processed_image)
                logger.info(f"Successfully processed image: {caption[:100]}...")
                
            except Exception as e:
                logger.error(f"Failed to process image: {e}")
                continue
        
        return processed_images
    
    def process_pdf_complete(self, pdf_path: str) -> List[ProcessedImage]:
        """Complete pipeline: extract and process all images from PDF"""
        logger.info(f"Starting complete PDF processing: {pdf_path}")
        
        # Extract images
        extracted_images = self.extract_images_from_pdf(pdf_path)
        
        if not extracted_images:
            logger.warning("No images found in PDF")
            return []
        
        # Process images
        processed_images = self.process_images(extracted_images)
        
        logger.info(f"PDF processing complete. Processed {len(processed_images)} images")
        return processed_images
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string for web display"""
        try:
            buffer = io.BytesIO()
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize for web display if too large
            max_display_size = 800
            if max(image.size) > max_display_size:
                ratio = max_display_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            image.save(buffer, format='JPEG', quality=85, optimize=True)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_str}"
            
        except Exception as e:
            logger.error(f"Failed to convert image to base64: {e}")
            return ""

# Utility functions
def get_pdf_metadata(pdf_path: str) -> Dict:
    """Extract PDF metadata"""
    try:
        pdf_doc = fitz.open(pdf_path)
        metadata = pdf_doc.metadata
        page_count = pdf_doc.page_count
        file_size = os.path.getsize(pdf_path)
        pdf_doc.close()
        
        return {
            "title": metadata.get("title", "Unknown"),
            "author": metadata.get("author", "Unknown"),
            "subject": metadata.get("subject", ""),
            "creator": metadata.get("creator", ""),
            "producer": metadata.get("producer", ""),
            "creation_date": metadata.get("creationDate", ""),
            "modification_date": metadata.get("modDate", ""),
            "page_count": page_count,
            "file_size_mb": round(file_size / (1024 * 1024), 2)
        }
        
    except Exception as e:
        logger.error(f"Failed to extract PDF metadata: {e}")
        return {}

# Example usage and testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python pdf_processor.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Test the processor
    processor = PDFProcessor()
    results = processor.process_pdf_complete(pdf_path)
    
    print(f"\nProcessed {len(results)} images:")
    for result in results:
        print(f"Page {result.page_number}, Image {result.image_index}: {result.caption}")
        print(f"Confidence: {result.confidence:.2f}")
        print("-" * 80)