#!/usr/bin/env python3
"""
Enhanced Railway PostgreSQL Database Setup Script
Run this after creating PostgreSQL service in Railway to initialize all tables and seed data
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def setup_railway_database():
    """Complete Railway PostgreSQL database setup"""
    print("🗄️ Setting up Railway PostgreSQL Database...")
    print("=" * 60)
    
    # 1. Check DATABASE_URL
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL environment variable not found!")
        print("\n📋 To fix this:")
        print("1. Go to your Railway dashboard")
        print("2. Create a PostgreSQL service")
        print("3. Copy the DATABASE_URL from variables")
        print("4. Set it as environment variable")
        return False
    
    print(f"✅ DATABASE_URL found: {database_url[:50]}...")
    
    # 2. Test database connection
    print("\n🔌 Testing database connection...")
    try:
        import psycopg2
        from urllib.parse import urlparse
        
        # Parse database URL
        parsed = urlparse(database_url)
        conn = psycopg2.connect(
            host=parsed.hostname,
            port=parsed.port,
            database=parsed.path[1:],  # Remove leading slash
            user=parsed.username,
            password=parsed.password,
            sslmode='require'
        )
        conn.close()
        print("✅ Database connection successful!")
        
    except ImportError:
        print("⚠️ psycopg2 not installed, but continuing with SQLAlchemy...")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False
    
    # 3. Initialize database schema
    print("\n📋 Creating database tables...")
    try:
        from database.models import init_database, get_database_stats
        
        # Initialize all tables
        init_database(database_url)
        print("✅ All database tables created successfully!")
        
        # Get initial stats
        stats = get_database_stats()
        print(f"📊 Database initialized with {len(stats)} metric types")
        
    except Exception as e:
        print(f"❌ Database table creation failed: {e}")
        print(f"Error details: {str(e)}")
        return False
    
    # 4. Seed initial data
    print("\n🌱 Seeding initial data...")
    try:
        seed_initial_data(database_url)
        print("✅ Initial data seeded successfully!")
        
    except Exception as e:
        print(f"⚠️ Warning: Initial data seeding failed: {e}")
        print("Database tables created but no seed data added")
    
    # 5. Verify setup
    print("\n🔍 Verifying database setup...")
    try:
        verify_database_setup(database_url)
        print("✅ Database setup verification passed!")
        
    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        return False
    
    # Success summary
    print("\n" + "=" * 60)
    print("🎉 Railway PostgreSQL Database Setup Complete!")
    print("=" * 60)
    print("✅ Database tables created")
    print("✅ Initial data seeded")
    print("✅ Setup verified")
    print("\n🚀 Your Railway app can now use persistent database storage!")
    print("📊 Check your Railway dashboard to see the PostgreSQL service")
    
    return True

def seed_initial_data(database_url):
    """Seed initial data for the application"""
    from database.models import get_db_session, ContentRule, SystemMetrics
    
    # Content rules for different cultural contexts
    initial_rules = [
        {
            "rule_name": "Islamic Cultural Compliance - High Risk",
            "rule_type": "keyword",
            "category": "islamic_compliance",
            "severity": "high",
            "rule_pattern": "alcohol,beer,wine,pork,gambling,casino,nudity,explicit",
            "confidence_threshold": 0.8,
            "is_active": True,
            "created_by": "system"
        },
        {
            "rule_name": "Inappropriate Clothing Detection",
            "rule_type": "semantic",
            "category": "clothing_modesty",
            "severity": "medium",
            "rule_pattern": "revealing clothing,short dress,bikini,swimsuit,low cut",
            "confidence_threshold": 0.7,
            "is_active": True,
            "created_by": "system"
        },
        {
            "rule_name": "Mixed Gender Interactions",
            "rule_type": "pattern",
            "category": "social_interactions",
            "severity": "medium",
            "rule_pattern": "man.*woman|woman.*man|couple.*together|dating|romantic",
            "confidence_threshold": 0.6,
            "is_active": True,
            "created_by": "system"
        }
    ]
    
    # System metrics initialization
    initial_metrics = [
        {
            "metric_name": "system_startup",
            "metric_type": "counter",
            "metric_value": 1.0,
            "component": "database",
            "tags": {"event": "initialization", "version": "1.0"}
        },
        {
            "metric_name": "database_tables_created",
            "metric_type": "gauge",
            "metric_value": 8.0,  # Number of main tables
            "component": "database",
            "tags": {"setup": "initial"}
        }
    ]
    
    with get_db_session() as session:
        # Add content rules
        for rule_data in initial_rules:
            rule = ContentRule(**rule_data)
            session.add(rule)
        
        # Add system metrics
        for metric_data in initial_metrics:
            metric = SystemMetrics(**metric_data)
            session.add(metric)
        
        session.commit()
        print(f"✅ Added {len(initial_rules)} content rules")
        print(f"✅ Added {len(initial_metrics)} system metrics")

def verify_database_setup(database_url):
    """Verify that database setup was successful"""
    from database.models import get_db_session, Document, AnalysisResult, Violation, ContentRule
    from sqlalchemy import text
    
    with get_db_session() as session:
        # Test basic queries on all main tables
        tables_to_check = [
            ('documents', Document),
            ('analysis_results', AnalysisResult),
            ('violations', Violation),
            ('content_rules', ContentRule)
        ]
        
        for table_name, model_class in tables_to_check:
            count = session.query(model_class).count()
            print(f"  📊 {table_name}: {count} records")
        
        # Test database functions
        result = session.execute(text("SELECT version()")).fetchone()
        print(f"  🗄️ PostgreSQL Version: {result[0][:50]}...")
        
        # Test JSON column functionality
        test_json = session.execute(text("SELECT '{\"test\": true}'::json")).fetchone()
        print(f"  🔧 JSON Support: {'✅' if test_json else '❌'}")

def create_backup_script():
    """Create a backup script for the database"""
    backup_script = '''#!/bin/bash
# Railway PostgreSQL Backup Script
# Run this to create a backup of your Railway database

echo "🗄️ Creating Railway PostgreSQL backup..."

# Get database URL from Railway
if [ -z "$DATABASE_URL" ]; then
    echo "❌ DATABASE_URL not set"
    exit 1
fi

# Create backup filename with timestamp
BACKUP_FILE="railway_backup_$(date +%Y%m%d_%H%M%S).sql"

echo "📁 Creating backup: $BACKUP_FILE"

# Create backup using pg_dump
pg_dump "$DATABASE_URL" > "$BACKUP_FILE"

if [ $? -eq 0 ]; then
    echo "✅ Backup created successfully: $BACKUP_FILE"
    echo "📊 Backup size: $(du -h "$BACKUP_FILE" | cut -f1)"
else
    echo "❌ Backup failed"
    exit 1
fi
'''
    
    with open("backup_railway_db.sh", "w") as f:
        f.write(backup_script)
    
    # Make executable
    os.chmod("backup_railway_db.sh", 0o755)
    print("✅ Created backup_railway_db.sh script")

def main():
    """Main setup function"""
    print("🚀 DocShield Pro - Railway Database Setup")
    print("=" * 60)
    
    # Check if running in Railway environment
    if os.getenv("RAILWAY_ENVIRONMENT"):
        print("🚂 Running in Railway environment")
    else:
        print("💻 Running in local environment")
        print("⚠️ Make sure to set DATABASE_URL for Railway PostgreSQL")
    
    try:
        success = setup_railway_database()
        
        if success:
            # Create additional helpful scripts
            create_backup_script()
            
            print("\n📝 Additional files created:")
            print("  - backup_railway_db.sh (database backup script)")
            
            print("\n🔗 Useful Railway commands:")
            print("  railway login")
            print("  railway environment")
            print("  railway logs")
            print("  railway shell")
        
        return success
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)