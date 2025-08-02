#!/usr/bin/env python3
"""
Railway Deployment Helper Script
Automates the deployment process to Railway
"""
import os
import sys
import subprocess
import json
from pathlib import Path

def check_prerequisites():
    """Check if all prerequisites are met"""
    print("🔍 Checking deployment prerequisites...")
    
    # Check if Railway CLI is installed
    try:
        result = subprocess.run(['railway', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Railway CLI: {result.stdout.strip()}")
        else:
            print("❌ Railway CLI not found!")
            print("Install it with: npm install -g @railway/cli")
            return False
    except FileNotFoundError:
        print("❌ Railway CLI not found!")
        print("Install it with: npm install -g @railway/cli")
        return False
    
    # Check if logged in to Railway
    try:
        result = subprocess.run(['railway', 'whoami'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Railway authenticated: {result.stdout.strip()}")
        else:
            print("❌ Not logged in to Railway!")
            print("Run: railway login")
            return False
    except:
        print("❌ Railway authentication check failed!")
        return False
    
    # Check if requirements.txt exists
    if Path("requirements.txt").exists():
        print("✅ requirements.txt found")
    else:
        print("❌ requirements.txt not found!")
        print("Create requirements.txt with all dependencies")
        return False
    
    return True

def setup_railway_project():
    """Initialize or connect to Railway project"""
    print("\n🚂 Setting up Railway project...")
    
    # Check if already linked to a project
    if Path(".railway").exists():
        print("✅ Already linked to Railway project")
        return True
    
    # Initialize new project or link existing
    print("Choose an option:")
    print("1. Create new Railway project")
    print("2. Link to existing project")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Create new project
        try:
            result = subprocess.run(['railway', 'init'], capture_output=True, text=True, input='y\n')
            if result.returncode == 0:
                print("✅ New Railway project created")
                return True
            else:
                print(f"❌ Failed to create project: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error creating project: {e}")
            return False
    
    elif choice == "2":
        # Link existing project
        try:
            result = subprocess.run(['railway', 'link'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Linked to existing Railway project")
                return True
            else:
                print(f"❌ Failed to link project: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error linking project: {e}")
            return False
    
    else:
        print("❌ Invalid choice")
        return False

def setup_postgresql():
    """Add PostgreSQL service to Railway project"""
    print("\n🗄️ Setting up PostgreSQL database...")
    
    try:
        # Add PostgreSQL plugin
        result = subprocess.run(['railway', 'add', '--plugin', 'postgresql'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ PostgreSQL service added")
            print("⏳ Waiting for database to be ready...")
            
            # Wait a moment for the service to be ready
            import time
            time.sleep(10)
            
            return True
        else:
            print("ℹ️ PostgreSQL service may already exist")
            return True
            
    except Exception as e:
        print(f"❌ Error setting up PostgreSQL: {e}")
        return False

def set_environment_variables():
    """Set required environment variables"""
    print("\n🔧 Setting environment variables...")
    
    env_vars = {
        'PORT': '8080',
        'HOST': '0.0.0.0',
        'PYTHONPATH': '/app',
        'CUDA_VISIBLE_DEVICES': '',  # Force CPU to avoid GPU issues
        'DEBUG': 'false',
        'RAILWAY_ENVIRONMENT': 'production'
    }
    
    for key, value in env_vars.items():
        try:
            result = subprocess.run(['railway', 'variables', 'set', f'{key}={value}'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Set {key}={value}")
            else:
                print(f"⚠️ Warning: Could not set {key}")
        except Exception as e:
            print(f"⚠️ Warning: Error setting {key}: {e}")

def initialize_database():
    """Initialize the database tables"""
    print("\n📋 Initializing database...")
    
    try:
        # Run database initialization
        result = subprocess.run(['railway', 'run', 'python', 'setup_railway_db.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Database initialized successfully")
            print(result.stdout)
            return True
        else:
            print("⚠️ Database initialization had issues:")
            print(result.stderr)
            print("You may need to run this manually after deployment")
            return False
            
    except Exception as e:
        print(f"⚠️ Database initialization error: {e}")
        print("You can run 'railway run python setup_railway_db.py' manually after deployment")
        return False

def deploy_application():
    """Deploy the application to Railway"""
    print("\n🚀 Deploying application...")
    
    try:
        # Deploy with Railway
        result = subprocess.run(['railway', 'up', '--detach'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Application deployed successfully!")
            print(result.stdout)
            
            # Get deployment URL
            try:
                url_result = subprocess.run(['railway', 'domain'], 
                                          capture_output=True, text=True)
                if url_result.returncode == 0:
                    print(f"🌐 Application URL: {url_result.stdout.strip()}")
                else:
                    print("ℹ️ Getting domain URL...")
                    print("You can get the URL with: railway domain")
            except:
                print("ℹ️ Get your app URL with: railway domain")
            
            return True
        else:
            print(f"❌ Deployment failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Deployment error: {e}")
        return False

def show_deployment_status():
    """Show deployment status and helpful commands"""
    print("\n📊 Deployment Status & Next Steps:")
    print("=" * 50)
    
    try:
        # Show status
        result = subprocess.run(['railway', 'status'], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
    except:
        print("⚠️ Could not get status")
    
    print("\n🔧 Useful Railway Commands:")
    print("  railway logs          - View application logs")
    print("  railway shell         - Open shell in deployment")
    print("  railway domain        - Get deployment URL")
    print("  railway variables     - Manage environment variables")
    print("  railway status        - Check deployment status")
    print("  railway restart       - Restart the application")
    
    print("\n🗄️ Database Commands:")
    print("  railway run python setup_railway_db.py    - Initialize database")
    print("  railway run python migrate_database.py    - Run migrations")
    
    print("\n📱 Monitoring:")
    print("  Check Railway dashboard for metrics and logs")
    print("  Monitor at: https://railway.app/dashboard")

def main():
    """Main deployment function"""
    print("🚀 DocShield Pro - Railway Deployment")
    print("=" * 50)
    
    try:
        # Check prerequisites
        if not check_prerequisites():
            print("\n❌ Prerequisites not met. Please fix and try again.")
            return False
        
        # Setup Railway project
        if not setup_railway_project():
            print("\n❌ Railway project setup failed.")
            return False
        
        # Setup PostgreSQL
        if not setup_postgresql():
            print("\n❌ PostgreSQL setup failed.")
            return False
        
        # Set environment variables
        set_environment_variables()
        
        # Deploy application
        if not deploy_application():
            print("\n❌ Application deployment failed.")
            return False
        
        # Initialize database (optional, can be done manually)
        print("\n🔄 Attempting database initialization...")
        initialize_database()
        
        # Show status and next steps
        show_deployment_status()
        
        print("\n🎉 Deployment process completed!")
        print("🔗 Check your Railway dashboard for the live application")
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Deployment interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)