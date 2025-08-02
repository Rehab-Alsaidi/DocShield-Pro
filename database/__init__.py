# database/__init__.py
"""
Database Package

Contains database models, initialization scripts, and migration files
for storing content moderation results and metadata.
"""

try:
    from .models import init_database
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    def init_database():
        """Placeholder function when database is not available"""
        pass

__all__ = [
    'init_database',
    'DATABASE_AVAILABLE'
]