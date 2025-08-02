#!/usr/bin/env python3
"""
Database Migration Script for DocShield Pro
Handles database schema updates and data migrations
"""
import os
import sys
from pathlib import Path
from datetime import datetime

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def run_migration():
    """Run database migration"""
    print("🔄 Starting Database Migration...")
    print("=" * 50)
    
    # Check database URL
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL not found!")
        return False
    
    print(f"🗄️ Database: {database_url[:50]}...")
    
    try:
        from database.models import DatabaseManager, Base
        from sqlalchemy import text
        
        # Connect to database
        db = DatabaseManager(database_url)
        db.connect()
        
        print("✅ Connected to database")
        
        # Check existing tables
        with db.get_session() as session:
            # Get list of existing tables
            result = session.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)).fetchall()
            
            existing_tables = [row[0] for row in result]
            print(f"📊 Found {len(existing_tables)} existing tables")
            
            # Define expected tables
            expected_tables = [
                'documents',
                'analysis_results', 
                'violations',
                'processing_logs',
                'model_performance',
                'user_sessions',
                'system_metrics',
                'content_rules'
            ]
            
            # Check which tables need to be created
            missing_tables = [table for table in expected_tables if table not in existing_tables]
            
            if missing_tables:
                print(f"🔧 Creating {len(missing_tables)} missing tables...")
                print(f"   Missing: {', '.join(missing_tables)}")
                
                # Create missing tables
                Base.metadata.create_all(bind=db.engine)
                print("✅ Missing tables created")
            else:
                print("✅ All tables already exist")
            
            # Run any custom migrations
            run_custom_migrations(session)
            
        db.close_connection()
        
        print("✅ Migration completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return False

def run_custom_migrations(session):
    """Run custom migration scripts"""
    print("🔧 Running custom migrations...")
    
    # Migration 1: Add indexes for better performance
    try:
        from sqlalchemy import text
        
        # Add indexes if they don't exist
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status)",
            "CREATE INDEX IF NOT EXISTS idx_documents_upload_time ON documents(upload_timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_analysis_risk_level ON analysis_results(overall_risk_level)",
            "CREATE INDEX IF NOT EXISTS idx_violations_severity ON violations(severity)",
            "CREATE INDEX IF NOT EXISTS idx_violations_confidence ON violations(confidence)",
            "CREATE INDEX IF NOT EXISTS idx_processing_logs_timestamp ON processing_logs(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_content_rules_active ON content_rules(is_active)"
        ]
        
        for index_sql in indexes:
            try:
                session.execute(text(index_sql))
                print(f"  ✅ Created index: {index_sql.split()[-1]}")
            except Exception as e:
                print(f"  ⚠️ Index creation skipped: {e}")
        
        session.commit()
        print("✅ Performance indexes added")
        
    except Exception as e:
        print(f"⚠️ Index creation failed: {e}")
    
    # Migration 2: Update content rules if needed
    try:
        from database.models import ContentRule
        
        # Check if we have any content rules
        rule_count = session.query(ContentRule).count()
        
        if rule_count == 0:
            print("  🌱 Seeding default content rules...")
            seed_default_rules(session)
        else:
            print(f"  📊 Found {rule_count} existing content rules")
        
    except Exception as e:
        print(f"⚠️ Content rules migration failed: {e}")

def seed_default_rules(session):
    """Seed default content moderation rules"""
    from database.models import ContentRule
    
    default_rules = [
        {
            "rule_name": "Islamic Compliance - Alcohol",
            "rule_type": "keyword",
            "category": "haram_substances",
            "severity": "high",
            "rule_pattern": "alcohol,beer,wine,whiskey,vodka,champagne,cocktail,drunk,drinking,bar,pub,nightclub",
            "confidence_threshold": 0.8,
            "is_active": True,
            "created_by": "system",
            "rule_metadata": {"cultural_context": "islamic", "region": "middle_east"}
        },
        {
            "rule_name": "Islamic Compliance - Pork Products",
            "rule_type": "keyword", 
            "category": "haram_food",
            "severity": "high",
            "rule_pattern": "pork,pig,bacon,ham,sausage,pepperoni,lard",
            "confidence_threshold": 0.9,
            "is_active": True,
            "created_by": "system",
            "rule_metadata": {"cultural_context": "islamic", "region": "middle_east"}
        },
        {
            "rule_name": "Inappropriate Clothing",
            "rule_type": "semantic",
            "category": "clothing_modesty",
            "severity": "medium",
            "rule_pattern": "revealing,bikini,underwear,lingerie,short dress,low cut,cleavage,swimsuit",
            "confidence_threshold": 0.7,
            "is_active": True,
            "created_by": "system",
            "rule_metadata": {"cultural_context": "conservative", "region": "middle_east"}
        },
        {
            "rule_name": "Mixed Gender Social Interactions",
            "rule_type": "pattern",
            "category": "social_interactions", 
            "severity": "medium",
            "rule_pattern": "dating,boyfriend,girlfriend,romantic,kissing,hugging,couple together",
            "confidence_threshold": 0.6,
            "is_active": True,
            "created_by": "system",
            "rule_metadata": {"cultural_context": "islamic", "region": "middle_east"}
        },
        {
            "rule_name": "Gambling Content",
            "rule_type": "keyword",
            "category": "haram_activities",
            "severity": "high", 
            "rule_pattern": "gambling,casino,poker,betting,cards,dice,slot machine,lottery,roulette",
            "confidence_threshold": 0.8,
            "is_active": True,
            "created_by": "system",
            "rule_metadata": {"cultural_context": "islamic", "region": "middle_east"}
        },
        {
            "rule_name": "Non-Islamic Religious Content",
            "rule_type": "keyword",
            "category": "religious_content",
            "severity": "medium",
            "rule_pattern": "christmas,easter,halloween,santa,cross,church,bible,priest",
            "confidence_threshold": 0.6,
            "is_active": True,
            "created_by": "system",
            "rule_metadata": {"cultural_context": "islamic", "region": "middle_east"}
        }
    ]
    
    for rule_data in default_rules:
        rule = ContentRule(**rule_data)
        session.add(rule)
    
    session.commit()
    print(f"  ✅ Added {len(default_rules)} default content rules")

def check_database_health():
    """Check database health and performance"""
    print("\n🏥 Database Health Check...")
    
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ No database URL found")
        return False
    
    try:
        from database.models import get_database_stats
        from sqlalchemy import create_engine, text
        
        # Get database statistics
        stats = get_database_stats()
        
        print("📊 Database Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
        
        # Check database performance
        engine = create_engine(database_url)
        with engine.connect() as conn:
            # Check database size
            result = conn.execute(text("""
                SELECT pg_size_pretty(pg_database_size(current_database())) as size
            """)).fetchone()
            print(f"  Database size: {result[0]}")
            
            # Check connection count
            result = conn.execute(text("""
                SELECT count(*) as connections 
                FROM pg_stat_activity 
                WHERE datname = current_database()
            """)).fetchone()
            print(f"  Active connections: {result[0]}")
            
        engine.dispose()
        print("✅ Database health check completed")
        return True
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def main():
    """Main migration function"""
    print("🚀 DocShield Pro - Database Migration")
    print("=" * 50)
    
    try:
        # Run migration
        migration_success = run_migration()
        
        if migration_success:
            # Run health check
            health_check_success = check_database_health()
            
            if health_check_success:
                print("\n🎉 Migration and health check completed successfully!")
                return True
            else:
                print("\n⚠️ Migration completed but health check failed")
                return False
        else:
            print("\n❌ Migration failed")
            return False
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Migration interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)