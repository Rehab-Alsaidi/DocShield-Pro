# app/__init__.py
"""
PDF Content Moderator Application Package

Main application package containing Flask app configuration, 
API routes, and core application logic.
"""

try:
    from .config import app_config, model_config, db_config
except ImportError:
    # Fallback if config modules don't exist
    app_config = None
    model_config = None
    db_config = None

__version__ = "4.0.0-railway-optimized"
__author__ = "PDF Content Moderator Team"

__all__ = [
    'app_config', 
    'model_config',
    'db_config'
]