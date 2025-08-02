# Railway PostgreSQL Setup Guide

## 🗄️ Database Setup on Railway

### Step 1: Create PostgreSQL Service
1. Go to your Railway dashboard
2. Click "New Service" → "Database" → "PostgreSQL" 
3. Railway will automatically provision the database and set `DATABASE_URL`

### Step 2: Verify Environment Variables
Your Railway project should now have:
- `DATABASE_URL` - Automatically set by Railway PostgreSQL service
- `PORT` - Automatically set by Railway (usually 3000)
- `RAILWAY_ENVIRONMENT` - Automatically set to "production"

### Step 3: Deploy
The app will automatically:
- ✅ Detect PostgreSQL via `DATABASE_URL`
- ✅ Initialize database tables on startup
- ✅ Test database connection
- ✅ Save all processing results to PostgreSQL

### What the Database Stores:
- **Documents**: Uploaded PDF files metadata
- **Analysis Results**: Risk levels, confidence scores, processing stats
- **Violations**: Individual content violations with evidence
- **Logs**: Processing logs for debugging
- **Performance**: Model performance metrics

### Monitoring Database
- Visit `/debug` to see database status
- Visit `/health` to check database connection
- Check Railway logs for database initialization messages

### Troubleshooting
If database fails to initialize:
1. Check Railway logs for specific error messages
2. Verify PostgreSQL service is running in Railway dashboard
3. App will automatically fall back to file storage if database fails

## 🚀 Benefits of Database Storage
- **Persistent**: Data survives deployments
- **Analytics**: Track usage patterns and performance
- **Search**: Find past analyses by filename or content
- **Scaling**: Better performance with multiple users