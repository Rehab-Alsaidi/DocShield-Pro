#!/usr/bin/env python3
"""
Railway PostgreSQL Database Initialization Script
Run this once after creating PostgreSQL service in Railway
"""
import os
import sys
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def init_railway_database():
    """Initialize Railway PostgreSQL database"""
    print("🗄️ Initializing Railway PostgreSQL Database...")
    
    # Check if DATABASE_URL is set
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL not found!")
        print("Make sure you've created a PostgreSQL service in Railway")
        return False
    
    print(f"✅ DATABASE_URL found: {database_url[:50]}...")
    
    try:
        # Import database modules
        from database.models import init_database
        
        print("📋 Creating database tables...")
        init_database(database_url)
        
        print("✅ Railway PostgreSQL database initialized successfully!")
        print("🚀 Your app can now use persistent database storage")
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

if __name__ == "__main__":
    success = init_railway_database()
    sys.exit(0 if success else 1)