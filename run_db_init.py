#!/usr/bin/env python3
"""
Simple script to run database initialization on Railway PostgreSQL
"""
import os
import sys
import psycopg2
from pathlib import Path

def run_sql_file():
    """Run the Railway init SQL file"""
    
    # Check if we have database URL
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ DATABASE_URL not found!")
        print("Set it with your Railway PostgreSQL URL:")
        print("postgresql://postgres:zGAfgQBGnZtUHgmdDZSRnFoeMvQxxVlb@postgres.railway.internal:5432/railway")
        return False
    
    print(f"🗄️ Connecting to Railway PostgreSQL...")
    print(f"Database: {db_url[:50]}...")
    
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(db_url, sslmode='require')
        cur = conn.cursor()
        
        print("✅ Connected to database")
        
        # Read and execute SQL file
        sql_file = Path(__file__).parent / "database" / "init_railway.sql"
        
        if not sql_file.exists():
            print(f"❌ SQL file not found: {sql_file}")
            return False
        
        print(f"📋 Reading SQL file: {sql_file}")
        
        with open(sql_file, 'r') as f:
            sql_content = f.read()
        
        print("⚡ Executing SQL commands...")
        
        # Execute the SQL
        cur.execute(sql_content)
        conn.commit()
        
        print("✅ Database tables created successfully!")
        
        # Verify tables were created
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name
        """)
        
        tables = cur.fetchall()
        print(f"📊 Created {len(tables)} tables:")
        for table in tables:
            print(f"  ✓ {table[0]}")
        
        # Check content rules
        cur.execute("SELECT COUNT(*) FROM content_rules")
        rule_count = cur.fetchone()[0]
        print(f"📋 Inserted {rule_count} content rules")
        
        cur.close()
        conn.close()
        
        print("\n🎉 Database initialization completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

if __name__ == "__main__":
    # Set the DATABASE_URL for testing
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        os.environ["DATABASE_URL"] = "postgresql://postgres:zGAfgQBGnZtUHgmdDZSRnFoeMvQxxVlb@postgres.railway.internal:5432/railway"
    
    success = run_sql_file()
    sys.exit(0 if success else 1)