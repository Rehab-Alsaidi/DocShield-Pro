# app/__init__.py
"""
PDF Content Moderator Application Package

Main application package containing Flask app configuration, 
API routes, and core application logic.
"""

from .main import create_app
from .config import app_config, model_config, db_config

__version__ = "1.0.0"
__author__ = "PDF Content Moderator Team"

__all__ = [
    'create_app',
    'app_config', 
    'model_config',
    'db_config'
]