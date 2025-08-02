-- Railway PostgreSQL initialization script
-- Remove the \c command that doesn't work in Railway and fix for Railway environment

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

CREATE INDEX IF NOT EXISTS idx_analysis_results_document_id ON analysis_results(document_id);
CREATE INDEX IF NOT EXISTS idx_analysis_results_risk_level ON analysis_results(overall_risk_level);

CREATE INDEX IF NOT EXISTS idx_violations_document_id ON violations(document_id);
CREATE INDEX IF NOT EXISTS idx_violations_severity ON violations(severity);

-- Insert default content rules
INSERT INTO content_rules (rule_name, rule_type, category, severity, rule_pattern, confidence_threshold) VALUES
('Islamic Compliance - Alcohol', 'keyword', 'haram_substances', 'high', 'alcohol,beer,wine,whiskey,vodka,champagne,cocktail,drunk,drinking,bar,pub,nightclub', 0.8),
('Islamic Compliance - Pork', 'keyword', 'haram_food', 'high', 'pork,pig,bacon,ham,sausage,pepperoni,lard', 0.9),
('Inappropriate Clothing', 'keyword', 'clothing_modesty', 'medium', 'revealing,bikini,underwear,lingerie,short dress,low cut,cleavage,swimsuit', 0.7),
('Mixed Gender Interactions', 'keyword', 'social_interactions', 'medium', 'dating,boyfriend,girlfriend,romantic,kissing,hugging,couple together', 0.6),
('Gambling Content', 'keyword', 'haram_activities', 'high', 'gambling,casino,poker,betting,cards,dice,slot machine,lottery,roulette', 0.8),
('Non-Islamic Religious Content', 'keyword', 'religious_content', 'medium', 'christmas,easter,halloween,santa,cross,church,bible,priest', 0.6)
ON CONFLICT DO NOTHING;