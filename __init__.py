# __init__.py
"""
PDF Content Moderator

An AI-powered system for analyzing and moderating PDF content using 
advanced computer vision and natural language processing models.

Features:
- Multi-modal AI analysis (Florence-2, CLIP, NSFW detection)
- Comprehensive text and image content filtering
- Detailed violation reporting and risk assessment
- REST API for integration
- PDF and HTML report generation

Usage:
    from pdf_content_moderator import ContentModerator
    
    moderator = ContentModerator()
    result = moderator.moderate_pdf("document.pdf")
"""

__version__ = "1.0.0"
__author__ = "PDF Content Moderator Team"
__description__ = "AI-powered PDF content moderation system"

# Main imports for easy access
try:
    from core import ContentModerator, ModerationResult
    from app import create_app
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

__all__ = [
    '__version__',
    '__author__', 
    '__description__',
    'CORE_AVAILABLE'
]

if CORE_AVAILABLE:
    __all__.extend(['ContentModerator', 'ModerationResult', 'create_app'])