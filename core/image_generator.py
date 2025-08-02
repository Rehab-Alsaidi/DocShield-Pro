# core/image_generator.py
"""
AI Image Generation for Cultural Compliance
Generates culturally appropriate replacement images
"""

import os
import io
import base64
import requests
import json
from typing import Dict, List, Optional, Union, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from dataclasses import dataclass
import logging

from utils.logger import get_logger
logger = get_logger(__name__)

@dataclass
class GeneratedImage:
    """Generated image with metadata"""
    image: Image.Image
    prompt: str
    style: str
    cultural_context: str
    confidence: float
    generation_method: str
    image_base64: str

class CulturalImageGenerator:
    """Generates culturally appropriate images for Middle Eastern/Islamic contexts"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")  # For DALL-E
        self.huggingface_key = os.getenv("HUGGINGFACE_API_KEY")  # For Stable Diffusion
        self.available_models = self._check_available_models()
        
        # Enhanced generation statistics
        self.generation_stats = {
            "total_generated": 0,
            "successful_generations": 0,
            "fallback_generations": 0,
            "model_usage": {}
        }
        
        # Cultural prompt templates
        self.cultural_templates = {
            "professional_meeting": {
                "base_prompt": "Professional business meeting in modern office setting",
                "cultural_modifiers": [
                    "with modest professional attire",
                    "culturally diverse professionals",
                    "respectful business environment",
                    "appropriate formal dress code"
                ],
                "style_modifiers": "clean, professional, corporate, high quality"
            },
            "traditional_gathering": {
                "base_prompt": "Traditional cultural gathering with family",
                "cultural_modifiers": [
                    "modest traditional clothing",
                    "respectful family interaction",
                    "cultural celebration setting",
                    "appropriate social gathering"
                ],
                "style_modifiers": "warm, traditional, family-friendly, cultural"
            },
            "educational_setting": {
                "base_prompt": "Educational conference or learning environment",
                "cultural_modifiers": [
                    "professional academic setting",
                    "appropriate learning materials",
                    "respectful educational interaction",
                    "culturally sensitive environment"
                ],
                "style_modifiers": "educational, professional, clean, academic"
            },
            "beverage_service": {
                "base_prompt": "Traditional beverage service presentation",
                "cultural_modifiers": [
                    "Arabic coffee or tea service",
                    "traditional serving methods",
                    "elegant cultural presentation",
                    "appropriate hospitality setting"
                ],
                "style_modifiers": "elegant, traditional, cultural, professional"
            },
            "architectural_elements": {
                "base_prompt": "Beautiful architectural elements and design",
                "cultural_modifiers": [
                    "Islamic geometric patterns",
                    "traditional architectural features",
                    "cultural design elements",
                    "respectful artistic representation"
                ],
                "style_modifiers": "artistic, architectural, geometric, elegant"
            },
            "nature_landscape": {
                "base_prompt": "Beautiful natural landscape",
                "cultural_modifiers": [
                    "serene natural environment",
                    "peaceful outdoor setting",
                    "appropriate natural beauty",
                    "culturally neutral landscape"
                ],
                "style_modifiers": "natural, peaceful, beautiful, serene"
            }
        }
        
        logger.info(f"Cultural Image Generator initialized with {len(self.available_models)} available models")
    
    def _check_available_models(self) -> List[str]:
        """Check which image generation models are available"""
        available = []
        
        # Check for DALL-E API access
        if self.api_key:
            available.append("dall-e-3")
            available.append("dall-e-2")
        
        # Check for Hugging Face API access
        if self.huggingface_key:
            available.append("stable-diffusion-xl")
        
        # Always available - synthetic generation
        available.append("synthetic-professional")
        
        return available
    
    def generate_replacement_image(self, original_analysis: Dict, replacement_suggestion: Dict) -> GeneratedImage:
        """Generate a culturally appropriate replacement image"""
        
        try:
            # Determine the best generation approach
            replacement_type = replacement_suggestion.get("replacement_type", "general_appropriate")
            generation_prompt = replacement_suggestion.get("generation_prompt", "")
            
            # Select appropriate cultural template
            template = self._select_cultural_template(replacement_type, original_analysis)
            
            # Build enhanced prompt
            enhanced_prompt = self._build_enhanced_prompt(template, generation_prompt, replacement_suggestion)
            
            # Generate image using available method
            generated_image = self._generate_with_best_available_method(enhanced_prompt, template)
            
            if generated_image:
                return generated_image
            else:
                # Fallback to synthetic generation
                return self._generate_synthetic_professional_image(enhanced_prompt, template)
                
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            # Return fallback synthetic image
            return self._generate_synthetic_professional_image(
                "Professional appropriate content", 
                self.cultural_templates["professional_meeting"]
            )
    
    def _select_cultural_template(self, replacement_type: str, original_analysis: Dict) -> Dict:
        """Select the most appropriate cultural template"""
        
        # Map replacement types to templates
        template_mapping = {
            "beverage_alternative": "beverage_service",
            "entertainment_alternative": "educational_setting",
            "modest_attire": "professional_meeting",
            "cultural_gathering": "traditional_gathering",
            "general_appropriate": "professional_meeting"
        }
        
        template_key = template_mapping.get(replacement_type, "professional_meeting")
        
        # Further refine based on original analysis
        original_concepts = original_analysis.get("concept", "").lower()
        
        if "alcohol" in original_concepts or "drink" in original_concepts:
            template_key = "beverage_service"
        elif "gambling" in original_concepts or "casino" in original_concepts:
            template_key = "educational_setting"
        elif "party" in original_concepts or "gathering" in original_concepts:
            template_key = "traditional_gathering"
        elif "clothing" in original_concepts or "attire" in original_concepts:
            template_key = "professional_meeting"
        
        return self.cultural_templates[template_key]
    
    def _build_enhanced_prompt(self, template: Dict, base_prompt: str, replacement_suggestion: Dict) -> str:
        """Build enhanced prompt with cultural considerations"""
        
        # Start with template base prompt
        if base_prompt and len(base_prompt.strip()) > 0:
            enhanced_prompt = base_prompt
        else:
            enhanced_prompt = template["base_prompt"]
        
        # Add cultural modifiers
        cultural_modifiers = ", ".join(template["cultural_modifiers"])
        enhanced_prompt += f", {cultural_modifiers}"
        
        # Add style modifiers
        style_modifiers = template["style_modifiers"]
        enhanced_prompt += f", {style_modifiers}"
        
        # Add quality and cultural compliance terms
        enhanced_prompt += ", high quality, professional photography, culturally appropriate"
        
        # Ensure Middle Eastern/Islamic compliance
        enhanced_prompt += ", Middle Eastern cultural compliance, Islamic values appropriate"
        
        # Add negative prompt elements (what to avoid)
        enhanced_prompt += " | AVOID: alcohol, gambling, inappropriate clothing, explicit content, western holidays"
        
        return enhanced_prompt
    
    def _generate_with_best_available_method(self, prompt: str, template: Dict) -> Optional[GeneratedImage]:
        """Generate image using the best available method"""
        
        # Try DALL-E 3 first (highest quality)
        if "dall-e-3" in self.available_models:
            try:
                return self._generate_with_dalle3(prompt, template)
            except Exception as e:
                logger.warning(f"DALL-E 3 generation failed: {e}")
        
        # Try DALL-E 2
        if "dall-e-2" in self.available_models:
            try:
                return self._generate_with_dalle2(prompt, template)
            except Exception as e:
                logger.warning(f"DALL-E 2 generation failed: {e}")
        
        # Try Stable Diffusion XL
        if "stable-diffusion-xl" in self.available_models:
            try:
                return self._generate_with_stable_diffusion(prompt, template)
            except Exception as e:
                logger.warning(f"Stable Diffusion generation failed: {e}")
        
        return None
    
    def _generate_with_dalle3(self, prompt: str, template: Dict) -> GeneratedImage:
        """Generate image using DALL-E 3"""
        
        # Clean prompt for DALL-E (remove negative prompts)
        clean_prompt = prompt.split(" | AVOID:")[0].strip()
        
        # DALL-E 3 API call
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "dall-e-3",
            "prompt": clean_prompt,
            "size": "1024x1024",
            "quality": "hd",
            "style": "natural"
        }
        
        response = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers=headers,
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            image_url = result["data"][0]["url"]
            
            # Download and process image
            img_response = requests.get(image_url)
            image = Image.open(io.BytesIO(img_response.content))
            
            # Convert to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            return GeneratedImage(
                image=image,
                prompt=clean_prompt,
                style="photorealistic",
                cultural_context="Middle Eastern compliant",
                confidence=0.9,
                generation_method="DALL-E 3",
                image_base64=f"data:image/png;base64,{img_base64}"
            )
        else:
            raise Exception(f"DALL-E 3 API error: {response.status_code}")
    
    def _generate_with_dalle2(self, prompt: str, template: Dict) -> GeneratedImage:
        """Generate image using DALL-E 2"""
        
        # Similar to DALL-E 3 but with different parameters
        clean_prompt = prompt.split(" | AVOID:")[0].strip()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "dall-e-2",
            "prompt": clean_prompt,
            "size": "1024x1024",
            "n": 1
        }
        
        response = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers=headers,
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            image_url = result["data"][0]["url"]
            
            img_response = requests.get(image_url)
            image = Image.open(io.BytesIO(img_response.content))
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            return GeneratedImage(
                image=image,
                prompt=clean_prompt,
                style="artistic",
                cultural_context="Middle Eastern compliant",
                confidence=0.85,
                generation_method="DALL-E 2",
                image_base64=f"data:image/png;base64,{img_base64}"
            )
        else:
            raise Exception(f"DALL-E 2 API error: {response.status_code}")
    
    def _generate_with_stable_diffusion(self, prompt: str, template: Dict) -> GeneratedImage:
        """Generate image using Stable Diffusion XL via Hugging Face API"""
        
        # Split prompt and negative prompt
        parts = prompt.split(" | AVOID:")
        positive_prompt = parts[0].strip()
        negative_prompt = parts[1].strip() if len(parts) > 1 else "nsfw, explicit, inappropriate"
        
        API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        headers = {"Authorization": f"Bearer {self.huggingface_key}"}
        
        data = {
            "inputs": positive_prompt,
            "parameters": {
                "negative_prompt": negative_prompt,
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "width": 1024,
                "height": 1024
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=data, timeout=60)
        
        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content))
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            return GeneratedImage(
                image=image,
                prompt=positive_prompt,
                style="realistic",
                cultural_context="Middle Eastern compliant",
                confidence=0.8,
                generation_method="Stable Diffusion XL",
                image_base64=f"data:image/png;base64,{img_base64}"
            )
        else:
            raise Exception(f"Stable Diffusion API error: {response.status_code}")
    
    def _generate_synthetic_professional_image(self, prompt: str, template: Dict) -> GeneratedImage:
        """Generate a synthetic professional image as fallback"""
        
        # Create a professional-looking synthetic image
        width, height = 1024, 1024
        
        # Create base image with professional gradient
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Create professional gradient background
        for y in range(height):
            # Blue to light blue gradient
            r = int(240 + (255 - 240) * (y / height))
            g = int(245 + (255 - 245) * (y / height))
            b = int(255)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        
        # Add professional elements based on template type
        if "professional_meeting" in str(template):
            self._add_meeting_elements(draw, width, height)
        elif "beverage_service" in str(template):
            self._add_beverage_elements(draw, width, height)
        elif "educational_setting" in str(template):
            self._add_educational_elements(draw, width, height)
        elif "traditional_gathering" in str(template):
            self._add_cultural_elements(draw, width, height)
        else:
            self._add_generic_professional_elements(draw, width, height)
        
        # Add text overlay
        try:
            # Try to load a font
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 48)
                small_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Add title
            title = "Professional Content"
            bbox = draw.textbbox((0, 0), title, font=font)
            text_width = bbox[2] - bbox[0]
            x = (width - text_width) // 2
            y = height // 4
            
            # Add shadow
            draw.text((x + 2, y + 2), title, font=font, fill=(0, 0, 0, 128))
            draw.text((x, y), title, font=font, fill=(41, 74, 122))  # Professional blue
            
            # Add subtitle
            subtitle = "Culturally Appropriate Content"
            bbox = draw.textbbox((0, 0), subtitle, font=small_font)
            text_width = bbox[2] - bbox[0]
            x = (width - text_width) // 2
            y = height // 4 + 80
            
            draw.text((x, y), subtitle, font=small_font, fill=(74, 85, 104))
            
        except Exception as e:
            logger.warning(f"Could not add text overlay: {e}")
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return GeneratedImage(
            image=image,
            prompt=prompt,
            style="synthetic_professional",
            cultural_context="Middle Eastern compliant",
            confidence=0.7,
            generation_method="Synthetic Generation",
            image_base64=f"data:image/png;base64,{img_base64}"
        )
    
    def _add_meeting_elements(self, draw: ImageDraw.Draw, width: int, height: int):
        """Add professional meeting visual elements"""
        
        # Add table representation (rectangle)
        table_width = width // 2
        table_height = height // 8
        table_x = (width - table_width) // 2
        table_y = height * 2 // 3
        
        draw.rectangle([table_x, table_y, table_x + table_width, table_y + table_height], 
                      fill=(139, 69, 19), outline=(101, 67, 33))  # Brown table
        
        # Add chair representations (small rectangles)
        chair_width = 30
        chair_height = 40
        
        for i in range(4):
            chair_x = table_x + (i * table_width // 4) + 20
            chair_y = table_y - chair_height - 10
            draw.rectangle([chair_x, chair_y, chair_x + chair_width, chair_y + chair_height],
                          fill=(105, 105, 105), outline=(64, 64, 64))
    
    def _add_beverage_elements(self, draw: ImageDraw.Draw, width: int, height: int):
        """Add traditional beverage service elements"""
        
        # Add serving tray (oval)
        tray_width = width // 3
        tray_height = height // 6
        tray_x = (width - tray_width) // 2
        tray_y = height // 2
        
        draw.ellipse([tray_x, tray_y, tray_x + tray_width, tray_y + tray_height],
                    fill=(184, 134, 11), outline=(139, 69, 19))  # Golden tray
        
        # Add cups (small circles)
        cup_radius = 20
        for i in range(3):
            cup_x = tray_x + (i + 1) * tray_width // 4
            cup_y = tray_y + tray_height // 2
            draw.ellipse([cup_x - cup_radius, cup_y - cup_radius, 
                         cup_x + cup_radius, cup_y + cup_radius],
                        fill=(139, 69, 19), outline=(101, 67, 33))
    
    def _add_educational_elements(self, draw: ImageDraw.Draw, width: int, height: int):
        """Add educational setting elements"""
        
        # Add presentation screen (rectangle)
        screen_width = width // 2
        screen_height = height // 3
        screen_x = (width - screen_width) // 2
        screen_y = height // 6
        
        draw.rectangle([screen_x, screen_y, screen_x + screen_width, screen_y + screen_height],
                      fill=(248, 248, 255), outline=(105, 105, 105))
        
        # Add books (small rectangles)
        book_width = 40
        book_height = 60
        
        for i in range(3):
            book_x = width // 4 + (i * 60)
            book_y = height * 3 // 4
            color = [(220, 20, 60), (30, 144, 255), (50, 205, 50)][i]  # Different book colors
            draw.rectangle([book_x, book_y, book_x + book_width, book_y + book_height],
                          fill=color, outline=(64, 64, 64))
    
    def _add_cultural_elements(self, draw: ImageDraw.Draw, width: int, height: int):
        """Add cultural gathering elements"""
        
        # Add geometric pattern (simplified Islamic pattern)
        center_x, center_y = width // 2, height // 2
        
        # Draw concentric circles with geometric elements
        for radius in [100, 150, 200]:
            draw.ellipse([center_x - radius, center_y - radius,
                         center_x + radius, center_y + radius],
                        outline=(41, 74, 122), width=3)
        
        # Add simple geometric shapes
        for angle in range(0, 360, 45):
            import math
            x = center_x + int(120 * math.cos(math.radians(angle)))
            y = center_y + int(120 * math.sin(math.radians(angle)))
            draw.rectangle([x-10, y-10, x+10, y+10], fill=(184, 134, 11))
    
    def _add_generic_professional_elements(self, draw: ImageDraw.Draw, width: int, height: int):
        """Add generic professional elements"""
        
        # Add simple architectural elements
        # Building outline
        building_width = width // 3
        building_height = height // 2
        building_x = (width - building_width) // 2
        building_y = height // 3
        
        draw.rectangle([building_x, building_y, building_x + building_width, building_y + building_height],
                      outline=(105, 105, 105), width=3)
        
        # Add windows
        window_size = 30
        for row in range(3):
            for col in range(4):
                window_x = building_x + 30 + (col * 60)
                window_y = building_y + 30 + (row * 60)
                draw.rectangle([window_x, window_y, window_x + window_size, window_y + window_size],
                              fill=(135, 206, 235), outline=(105, 105, 105))
    
    def generate_before_after_comparison(self, original_image: Image.Image, 
                                       generated_image: GeneratedImage) -> Image.Image:
        """Create a before/after comparison image"""
        
        # Resize images to same size
        target_size = (512, 512)
        original_resized = original_image.resize(target_size)
        generated_resized = generated_image.image.resize(target_size)
        
        # Create comparison image
        comparison_width = target_size[0] * 2 + 60  # Gap between images
        comparison_height = target_size[1] + 120  # Space for labels
        
        comparison = Image.new('RGB', (comparison_width, comparison_height), color='white')
        
        # Paste images
        comparison.paste(original_resized, (20, 60))
        comparison.paste(generated_resized, (target_size[0] + 40, 60))
        
        # Add labels
        draw = ImageDraw.Draw(comparison)
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            large_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 32)
        except:
            font = ImageFont.load_default()
            large_font = ImageFont.load_default()
        
        # Title
        title = "Content Replacement Comparison"
        bbox = draw.textbbox((0, 0), title, font=large_font)
        title_width = bbox[2] - bbox[0]
        title_x = (comparison_width - title_width) // 2
        draw.text((title_x, 10), title, font=large_font, fill=(41, 74, 122))
        
        # Labels
        draw.text((20 + target_size[0]//2 - 50, 30), "Original", font=font, fill=(220, 20, 60))
        draw.text((target_size[0] + 40 + target_size[0]//2 - 70, 30), "Replacement", font=font, fill=(50, 205, 50))
        
        # Status
        status_y = target_size[1] + 80
        draw.text((20, status_y), "❌ Cultural compliance issue", font=font, fill=(220, 20, 60))
        draw.text((target_size[0] + 40, status_y), "✅ Culturally appropriate", font=font, fill=(50, 205, 50))
        
        return comparison
    
    def batch_generate_replacements(self, analysis_results: List[Dict], 
                                  replacement_suggestions: List[Dict]) -> List[GeneratedImage]:
        """Generate multiple replacement images in batch"""
        
        generated_images = []
        
        for i, (analysis, suggestion) in enumerate(zip(analysis_results, replacement_suggestions)):
            try:
                logger.info(f"Generating replacement image {i+1}/{len(analysis_results)}")
                generated_image = self.generate_replacement_image(analysis, suggestion)
                generated_images.append(generated_image)
            except Exception as e:
                logger.error(f"Failed to generate replacement image {i+1}: {e}")
                # Add fallback synthetic image
                fallback = self._generate_synthetic_professional_image(
                    "Professional appropriate content",
                    self.cultural_templates["professional_meeting"]
                )
                generated_images.append(fallback)
        
        logger.info(f"Generated {len(generated_images)} replacement images")
        return generated_images

# Example usage and testing
if __name__ == "__main__":
    # Initialize generator
    generator = CulturalImageGenerator()
    
    print(f"Available models: {generator.available_models}")
    
    # Test synthetic generation
    test_analysis = {
        "concept": "alcohol bottles in party scene",
        "category": "prohibited_high",
        "severity": "high"
    }
    
    test_suggestion = {
        "replacement_type": "beverage_alternative",
        "generation_prompt": "Traditional Arabic coffee service in professional setting",
        "priority": "high"
    }
    
    print("Generating test replacement image...")
    result = generator.generate_replacement_image(test_analysis, test_suggestion)
    
    print(f"Generated image using: {result.generation_method}")
    print(f"Cultural context: {result.cultural_context}")
    print(f"Confidence: {result.confidence}")
    print(f"Image size: {result.image.size}")
    
    # Save for testing
    result.image.save("test_generated_image.png")
    print("Test image saved as 'test_generated_image.png'")