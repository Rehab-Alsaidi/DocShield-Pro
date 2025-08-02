-- database/init.sql
-- PostgreSQL initialization script for PDF Content Moderator

-- Create database (run as superuser)
-- CREATE DATABASE content_moderator;
-- CREATE USER moderator_user WITH PASSWORD 'secure_password_change_in_production';
-- GRANT ALL PRIVILEGES ON DATABASE content_moderator TO moderator_user;

-- Connect to the content_moderator database
\c content_moderator;

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create custom types
DO $$ BEGIN
    CREATE TYPE processing_status AS ENUM ('pending', 'processing', 'completed', 'failed');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE risk_level AS ENUM ('high', 'medium', 'low', 'unknown');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE violation_type AS ENUM ('image', 'text', 'combined');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE log_level AS ENUM ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_size INTEGER NOT NULL,
    file_hash VARCHAR(64),
    mime_type VARCHAR(100) DEFAULT 'application/pdf',
    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_status processing_status DEFAULT 'pending',
    file_path VARCHAR(500),
    results_path VARCHAR(500),
    report_path VARCHAR(500),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Analysis results table
CREATE TABLE IF NOT EXISTS analysis_results (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    document_id VARCHAR(36) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    overall_risk_level risk_level NOT NULL,
    overall_confidence FLOAT NOT NULL CHECK (overall_confidence >= 0 AND overall_confidence <= 1),
    total_violations INTEGER DEFAULT 0,
    total_pages INTEGER DEFAULT 0,
    total_images INTEGER DEFAULT 0,
    processing_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_time_seconds FLOAT,
    models_used JSONB,
    processing_device VARCHAR(50),
    summary_stats JSONB,
    processing_metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Violations table
CREATE TABLE IF NOT EXISTS violations (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    document_id VARCHAR(36) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    violation_type violation_type NOT NULL,
    page_number INTEGER NOT NULL,
    severity risk_level NOT NULL,
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    category VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    evidence JSONB,
    risk_factors JSONB,
    detected_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Processing logs table
CREATE TABLE IF NOT EXISTS processing_logs (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    document_id VARCHAR(36) REFERENCES documents(id) ON DELETE CASCADE,
    log_level log_level NOT NULL,
    message TEXT NOT NULL,
    component VARCHAR(100),
    function_name VARCHAR(100),
    context_data JSONB,
    error_details JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model performance table
CREATE TABLE IF NOT EXISTS model_performance (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    model_name VARCHAR(200) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    processing_time FLOAT NOT NULL,
    memory_usage_mb FLOAT,
    accuracy_score FLOAT CHECK (accuracy_score >= 0 AND accuracy_score <= 1),
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 1),
    input_size INTEGER,
    device_used VARCHAR(50),
    batch_size INTEGER,
    performance_metadata JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User sessions table
CREATE TABLE IF NOT EXISTS user_sessions (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    session_id VARCHAR(100) NOT NULL UNIQUE,
    ip_address INET,
    user_agent VARCHAR(500),
    documents_processed INTEGER DEFAULT 0,
    total_processing_time FLOAT DEFAULT 0.0,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    metric_name VARCHAR(100) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    metric_value FLOAT NOT NULL,
    component VARCHAR(100),
    tags JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Content rules table
CREATE TABLE IF NOT EXISTS content_rules (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    rule_name VARCHAR(100) NOT NULL,
    rule_type VARCHAR(50) NOT NULL,
    category VARCHAR(100) NOT NULL,
    severity risk_level NOT NULL,
    rule_pattern TEXT NOT NULL,
    confidence_threshold FLOAT DEFAULT 0.7 CHECK (confidence_threshold >= 0 AND confidence_threshold <= 1),
    is_active BOOLEAN DEFAULT TRUE,
    created_by VARCHAR(100),
    rule_metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status);
CREATE INDEX IF NOT EXISTS idx_documents_upload_time ON documents(upload_timestamp);
CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);
CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);

CREATE INDEX IF NOT EXISTS idx_analysis_results_document_id ON analysis_results(document_id);
CREATE INDEX IF NOT EXISTS idx_analysis_results_risk_level ON analysis_results(overall_risk_level);
CREATE INDEX IF NOT EXISTS idx_analysis_results_timestamp ON analysis_results(processing_timestamp);

CREATE INDEX IF NOT EXISTS idx_violations_document_id ON violations(document_id);
CREATE INDEX IF NOT EXISTS idx_violations_severity ON violations(severity);
CREATE INDEX IF NOT EXISTS idx_violations_category ON violations(category);
CREATE INDEX IF NOT EXISTS idx_violations_page ON violations(page_number);
CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON violations(detected_timestamp);

CREATE INDEX IF NOT EXISTS idx_processing_logs_document_id ON processing_logs(document_id);
CREATE INDEX IF NOT EXISTS idx_processing_logs_level ON processing_logs(log_level);
CREATE INDEX IF NOT EXISTS idx_processing_logs_component ON processing_logs(component);
CREATE INDEX IF NOT EXISTS idx_processing_logs_timestamp ON processing_logs(timestamp);

CREATE INDEX IF NOT EXISTS idx_model_performance_name ON model_performance(model_name);
CREATE INDEX IF NOT EXISTS idx_model_performance_type ON model_performance(model_type);
CREATE INDEX IF NOT EXISTS idx_model_performance_timestamp ON model_performance(timestamp);

CREATE INDEX IF NOT EXISTS idx_user_sessions_session_id ON user_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_last_activity ON user_sessions(last_activity);

CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON system_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_system_metrics_component ON system_metrics(component);
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);

CREATE INDEX IF NOT EXISTS idx_content_rules_category ON content_rules(category);
CREATE INDEX IF NOT EXISTS idx_content_rules_active ON content_rules(is_active);
CREATE INDEX IF NOT EXISTS idx_content_rules_severity ON content_rules(severity);

-- Create composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_violations_doc_severity ON violations(document_id, severity);
CREATE INDEX IF NOT EXISTS idx_analysis_risk_confidence ON analysis_results(overall_risk_level, overall_confidence);
CREATE INDEX IF NOT EXISTS idx_documents_status_time ON documents(processing_status, upload_timestamp);

-- Create JSONB indexes for metadata queries
CREATE INDEX IF NOT EXISTS idx_documents_metadata_gin ON documents USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_analysis_summary_stats_gin ON analysis_results USING GIN (summary_stats);
CREATE INDEX IF NOT EXISTS idx_violations_evidence_gin ON violations USING GIN (evidence);

-- Create triggers for updating timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update triggers
CREATE TRIGGER update_documents_updated_at 
    BEFORE UPDATE ON documents 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_content_rules_updated_at 
    BEFORE UPDATE ON content_rules 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries
CREATE OR REPLACE VIEW document_analysis_view AS
SELECT 
    d.id,
    d.filename,
    d.original_filename,
    d.file_size,
    d.upload_timestamp,
    d.processing_status,
    ar.overall_risk_level,
    ar.overall_confidence,
    ar.total_violations,
    ar.total_pages,
    ar.total_images,
    ar.processing_timestamp,
    ar.processing_time_seconds
FROM documents d
LEFT JOIN analysis_results ar ON d.id = ar.document_id;

CREATE OR REPLACE VIEW violation_summary_view AS
SELECT 
    document_id,
    COUNT(*) as total_violations,
    COUNT(CASE WHEN severity = 'high' THEN 1 END) as high_severity,
    COUNT(CASE WHEN severity = 'medium' THEN 1 END) as medium_severity,
    COUNT(CASE WHEN severity = 'low' THEN 1 END) as low_severity,
    AVG(confidence) as avg_confidence,
    STRING_AGG(DISTINCT category, ', ') as categories
FROM violations
GROUP BY document_id;

CREATE OR REPLACE VIEW daily_processing_stats AS
SELECT 
    DATE(upload_timestamp) as processing_date,
    COUNT(*) as documents_processed,
    COUNT(CASE WHEN processing_status = 'completed' THEN 1 END) as completed,
    COUNT(CASE WHEN processing_status = 'failed' THEN 1 END) as failed,
    AVG(CASE WHEN ar.processing_time_seconds IS NOT NULL THEN ar.processing_time_seconds END) as avg_processing_time,
    SUM(file_size) / 1024 / 1024 as total_mb_processed
FROM documents d
LEFT JOIN analysis_results ar ON d.id = ar.document_id
GROUP BY DATE(upload_timestamp)
ORDER BY processing_date DESC;

-- Insert default content rules
INSERT INTO content_rules (rule_name, rule_type, category, severity, rule_pattern, confidence_threshold) VALUES
-- High risk rules
('Explicit Content Keywords', 'keyword', 'adult_content', 'high', 'nude,naked,explicit,sexual,intimate,porn,xxx', 0.8),
('Violence Keywords', 'keyword', 'violence', 'high', 'violence,weapon,gun,knife,blood,fighting,war,kill', 0.7),
('Drug References', 'keyword', 'alcohol_drugs', 'high', 'drugs,cocaine,heroin,marijuana,pills,substance', 0.8),

-- Medium risk rules  
('Alcohol References', 'keyword', 'alcohol_drugs', 'medium', 'alcohol,beer,wine,drinking,drunk,bar,cocktail', 0.6),
('Gambling Content', 'keyword', 'gambling', 'medium', 'casino,gambling,poker,betting,cards,dice,lottery', 0.7),
('Inappropriate Clothing', 'keyword', 'inappropriate_clothing', 'medium', 'revealing,bikini,underwear,lingerie,swimsuit', 0.6),

-- Low risk rules
('Dating Content', 'keyword', 'dating_romance', 'low', 'dating,kissing,couple,romantic,wedding,marriage', 0.5),
('Western Holidays', 'keyword', 'western_holidays', 'low', 'christmas,easter,halloween,valentine,santa', 0.6),
('Religious Content', 'keyword', 'religious_content', 'low', 'church,mosque,temple,cross,prayer,worship', 0.5)

ON CONFLICT DO NOTHING;

-- Create stored procedures for common operations
CREATE OR REPLACE FUNCTION get_document_summary(doc_id VARCHAR(36))
RETURNS TABLE (
    document_id VARCHAR(36),
    filename VARCHAR(255),
    risk_level risk_level,
    violation_count INTEGER,
    processing_time FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id,
        d.filename,
        ar.overall_risk_level,
        ar.total_violations,
        ar.processing_time_seconds
    FROM documents d
    LEFT JOIN analysis_results ar ON d.id = ar.document_id
    WHERE d.id = doc_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION cleanup_old_records(days_old INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
    cutoff_date TIMESTAMP;
BEGIN
    cutoff_date := CURRENT_TIMESTAMP - INTERVAL '1 day' * days_old;
    
    -- Delete old processing logs
    DELETE FROM processing_logs WHERE timestamp < cutoff_date;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Delete old system metrics
    DELETE FROM system_metrics WHERE timestamp < cutoff_date;
    
    -- Delete old user sessions
    DELETE FROM user_sessions WHERE last_activity < cutoff_date;
    
    -- Delete completed documents older than retention period
    DELETE FROM documents 
    WHERE processing_status = 'completed' 
    AND upload_timestamp < cutoff_date;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO moderator_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO moderator_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO moderator_user;

-- Create health check function
CREATE OR REPLACE FUNCTION health_check()
RETURNS TABLE (
    check_name TEXT,
    status TEXT,
    details JSONB
) AS $$
BEGIN
    -- Check if tables exist and are accessible
    RETURN QUERY
    SELECT 
        'database_connectivity'::TEXT,
        'healthy'::TEXT,
        jsonb_build_object(
            'timestamp', CURRENT_TIMESTAMP,
            'document_count', (SELECT COUNT(*) FROM documents),
            'analysis_count', (SELECT COUNT(*) FROM analysis_results)
        );
    
    -- Add more health checks as needed
    RETURN QUERY
    SELECT 
        'table_sizes'::TEXT,
        'info'::TEXT,
        jsonb_build_object(
            'documents', (SELECT COUNT(*) FROM documents),
            'violations', (SELECT COUNT(*) FROM violations),
            'processing_logs', (SELECT COUNT(*) FROM processing_logs)
        );
END;
$$ LANGUAGE plpgsql;

-- Create indexes on JSONB fields for better query performance
CREATE INDEX IF NOT EXISTS idx_metadata_file_type ON documents USING GIN ((metadata->>'file_type'));
CREATE INDEX IF NOT EXISTS idx_summary_violation_types ON analysis_results USING GIN ((summary_stats->'violation_stats'));

-- Add comments for documentation
COMMENT ON TABLE documents IS 'Stores uploaded PDF documents and their metadata';
COMMENT ON TABLE analysis_results IS 'Stores the results of content moderation analysis';
COMMENT ON TABLE violations IS 'Individual content violations found during analysis';
COMMENT ON TABLE processing_logs IS 'System logs for debugging and monitoring';
COMMENT ON TABLE model_performance IS 'Performance metrics for AI models';
COMMENT ON TABLE user_sessions IS 'User session tracking for analytics';
COMMENT ON TABLE system_metrics IS 'System-wide metrics and statistics';
COMMENT ON TABLE content_rules IS 'Configurable content moderation rules';

COMMENT ON FUNCTION health_check() IS 'Returns database health status for monitoring';
COMMENT ON FUNCTION cleanup_old_records(INTEGER) IS 'Removes old records to maintain database size';
COMMENT ON FUNCTION get_document_summary(VARCHAR) IS 'Returns summary information for a specific document';

-- Final setup message
DO $$
BEGIN
    RAISE NOTICE 'PDF Content Moderator database initialized successfully!';
    RAISE NOTICE 'Database includes: % tables, % indexes, % functions, % views', 
        (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'),
        (SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public'),
        (SELECT COUNT(*) FROM information_schema.routines WHERE routine_schema = 'public'),
        (SELECT COUNT(*) FROM information_schema.views WHERE table_schema = 'public');
END $$;