# database/models.py
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import uuid
from typing import Optional, Dict, Any

from app.config import db_config
from utils.logger import get_logger

logger = get_logger(__name__)

# Create declarative base
Base = declarative_base()

class Document(Base):
    """Document table for tracking uploaded PDFs"""
    __tablename__ = 'documents'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)  # Size in bytes
    file_hash = Column(String(64), nullable=True)  # SHA256 hash
    mime_type = Column(String(100), default='application/pdf')
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    processing_status = Column(String(50), default='pending')  # pending, processing, completed, failed
    
    # File paths
    file_path = Column(String(500), nullable=True)
    results_path = Column(String(500), nullable=True)
    report_path = Column(String(500), nullable=True)
    
    # Metadata
    document_metadata = Column(JSON, nullable=True)
    
    # Relationships
    analysis_results = relationship("AnalysisResult", back_populates="document", cascade="all, delete-orphan")
    violations = relationship("Violation", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id='{self.id}', filename='{self.filename}', status='{self.processing_status}')>"

class AnalysisResult(Base):
    """Analysis results for documents"""
    __tablename__ = 'analysis_results'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey('documents.id'), nullable=False)
    
    # Overall results
    overall_risk_level = Column(String(20), nullable=False)  # high, medium, low
    overall_confidence = Column(Float, nullable=False)
    total_violations = Column(Integer, default=0)
    total_pages = Column(Integer, default=0)
    total_images = Column(Integer, default=0)
    
    # Processing information
    processing_timestamp = Column(DateTime, default=datetime.utcnow)
    processing_time_seconds = Column(Float, nullable=True)
    models_used = Column(JSON, nullable=True)  # List of model names
    processing_device = Column(String(50), nullable=True)
    
    # Summary statistics
    summary_stats = Column(JSON, nullable=True)
    processing_metadata = Column(JSON, nullable=True)
    
    # Relationships
    document = relationship("Document", back_populates="analysis_results")
    
    def __repr__(self):
        return f"<AnalysisResult(id='{self.id}', risk_level='{self.overall_risk_level}', violations={self.total_violations})>"

class Violation(Base):
    """Individual content violations"""
    __tablename__ = 'violations'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey('documents.id'), nullable=False)
    
    # Violation details
    violation_type = Column(String(50), nullable=False)  # image, text, combined
    page_number = Column(Integer, nullable=False)
    severity = Column(String(20), nullable=False)  # high, medium, low
    confidence = Column(Float, nullable=False)
    category = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    
    # Evidence and context
    evidence = Column(JSON, nullable=True)
    risk_factors = Column(JSON, nullable=True)  # List of risk factors
    
    # Timestamps
    detected_timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="violations")
    
    def __repr__(self):
        return f"<Violation(id='{self.id}', type='{self.violation_type}', severity='{self.severity}')>"

class ProcessingLog(Base):
    """Log of processing activities"""
    __tablename__ = 'processing_logs'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey('documents.id'), nullable=True)
    
    # Log details
    log_level = Column(String(20), nullable=False)  # INFO, WARNING, ERROR, DEBUG
    message = Column(Text, nullable=False)
    component = Column(String(100), nullable=True)  # pdf_processor, vision_analyzer, etc.
    function_name = Column(String(100), nullable=True)
    
    # Context
    context_data = Column(JSON, nullable=True)
    error_details = Column(JSON, nullable=True)
    
    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ProcessingLog(id='{self.id}', level='{self.log_level}', component='{self.component}')>"

class ModelPerformance(Base):
    """Track model performance metrics"""
    __tablename__ = 'model_performance'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Model information
    model_name = Column(String(200), nullable=False)
    model_type = Column(String(100), nullable=False)  # vision, nlp, classification
    model_version = Column(String(50), nullable=True)
    
    # Performance metrics
    processing_time = Column(Float, nullable=False)
    memory_usage_mb = Column(Float, nullable=True)
    accuracy_score = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Context
    input_size = Column(Integer, nullable=True)  # Image size, text length, etc.
    device_used = Column(String(50), nullable=True)
    batch_size = Column(Integer, nullable=True)
    
    # Additional metrics
    performance_metadata = Column(JSON, nullable=True)
    
    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ModelPerformance(model='{self.model_name}', time={self.processing_time:.3f}s)>"

class UserSession(Base):
    """Track user sessions for analytics"""
    __tablename__ = 'user_sessions'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Session information
    session_id = Column(String(100), nullable=False, unique=True)
    ip_address = Column(String(45), nullable=True)  # Support IPv6
    user_agent = Column(String(500), nullable=True)
    
    # Activity tracking
    documents_processed = Column(Integer, default=0)
    total_processing_time = Column(Float, default=0.0)
    last_activity = Column(DateTime, default=datetime.utcnow)
    
    # Session metadata
    session_metadata = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<UserSession(id='{self.session_id}', docs={self.documents_processed})>"

class SystemMetrics(Base):
    """System-wide metrics and statistics"""
    __tablename__ = 'system_metrics'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Metric information
    metric_name = Column(String(100), nullable=False)
    metric_type = Column(String(50), nullable=False)  # counter, gauge, histogram
    metric_value = Column(Float, nullable=False)
    
    # Context
    component = Column(String(100), nullable=True)
    tags = Column(JSON, nullable=True)  # Additional tags for filtering
    
    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<SystemMetrics(name='{self.metric_name}', value={self.metric_value})>"

class ContentRule(Base):
    """Configurable content moderation rules"""
    __tablename__ = 'content_rules'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Rule information
    rule_name = Column(String(100), nullable=False)
    rule_type = Column(String(50), nullable=False)  # keyword, pattern, semantic
    category = Column(String(100), nullable=False)
    severity = Column(String(20), nullable=False)  # high, medium, low
    
    # Rule definition
    rule_pattern = Column(Text, nullable=False)  # Keyword, regex, or semantic description
    confidence_threshold = Column(Float, default=0.7)
    
    # Status
    is_active = Column(Boolean, default=True)
    created_by = Column(String(100), nullable=True)
    
    # Metadata
    rule_metadata = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<ContentRule(name='{self.rule_name}', category='{self.category}', severity='{self.severity}')>"

# Database connection and session management
class DatabaseManager:
    """Database connection and session management"""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or db_config.database_url
        self.engine = None
        self.SessionLocal = None
        self.connected = False
        
    def connect(self):
        """Establish database connection"""
        try:
            # Get database configuration
            connect_args = db_config.connect_args or {}
            
            self.engine = create_engine(
                self.database_url,
                pool_size=db_config.pool_size,
                max_overflow=db_config.max_overflow,
                pool_timeout=db_config.pool_timeout,
                pool_recycle=db_config.pool_recycle,
                connect_args=connect_args,
                echo=False  # Set to True for SQL debugging
            )
            
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Test connection
            with self.engine.connect() as conn:
                from sqlalchemy import text
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            self.connected = True
            logger.info("Database connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def create_tables(self):
        """Create all tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def get_session(self):
        """Get database session"""
        if not self.connected:
            raise RuntimeError("Database not connected")
        
        return self.SessionLocal()
    
    def close_connection(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            self.connected = False
            logger.info("Database connection closed")

# Context manager for database sessions
from contextlib import contextmanager

@contextmanager
def get_db_session():
    """Context manager for database sessions"""
    db = DatabaseManager()
    db.connect()
    session = db.get_session()
    
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()
        db.close_connection()

# Data Access Layer (DAL)
class DocumentDAL:
    """Data Access Layer for Document operations"""
    
    def __init__(self, session):
        self.session = session
    
    def create_document(self, filename: str, original_filename: str, 
                       file_size: int, file_path: str, **kwargs) -> Document:
        """Create new document record"""
        doc = Document(
            filename=filename,
            original_filename=original_filename,
            file_size=file_size,
            file_path=file_path,
            **kwargs
        )
        
        self.session.add(doc)
        self.session.flush()  # Get the ID without committing
        return doc
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID"""
        return self.session.query(Document).filter(Document.id == document_id).first()
    
    def update_document_status(self, document_id: str, status: str):
        """Update document processing status"""
        doc = self.get_document(document_id)
        if doc:
            doc.processing_status = status
            self.session.flush()
    
    def delete_document(self, document_id: str) -> bool:
        """Delete document and all related records"""
        doc = self.get_document(document_id)
        if doc:
            self.session.delete(doc)
            return True
        return False

class AnalysisResultDAL:
    """Data Access Layer for Analysis Results"""
    
    def __init__(self, session):
        self.session = session
    
    def create_analysis_result(self, document_id: str, overall_risk_level: str,
                              overall_confidence: float, **kwargs) -> AnalysisResult:
        """Create analysis result record"""
        result = AnalysisResult(
            document_id=document_id,
            overall_risk_level=overall_risk_level,
            overall_confidence=overall_confidence,
            **kwargs
        )
        
        self.session.add(result)
        self.session.flush()
        return result
    
    def get_analysis_result(self, document_id: str) -> Optional[AnalysisResult]:
        """Get analysis result by document ID"""
        return self.session.query(AnalysisResult).filter(
            AnalysisResult.document_id == document_id
        ).first()

class ViolationDAL:
    """Data Access Layer for Violations"""
    
    def __init__(self, session):
        self.session = session
    
    def create_violation(self, document_id: str, violation_type: str,
                        page_number: int, severity: str, confidence: float,
                        category: str, description: str, **kwargs) -> Violation:
        """Create violation record"""
        violation = Violation(
            document_id=document_id,
            violation_type=violation_type,
            page_number=page_number,
            severity=severity,
            confidence=confidence,
            category=category,
            description=description,
            **kwargs
        )
        
        self.session.add(violation)
        self.session.flush()
        return violation
    
    def get_violations_by_document(self, document_id: str) -> list:
        """Get all violations for a document"""
        return self.session.query(Violation).filter(
            Violation.document_id == document_id
        ).order_by(Violation.severity.desc(), Violation.confidence.desc()).all()

# Utility functions
def init_database(database_url: Optional[str] = None):
    """Initialize database with tables"""
    db = DatabaseManager(database_url)
    db.connect()
    db.create_tables()
    db.close_connection()
    logger.info("Database initialized successfully")

def get_database_stats() -> Dict[str, Any]:
    """Get database statistics"""
    try:
        with get_db_session() as session:
            stats = {
                'total_documents': session.query(Document).count(),
                'completed_analyses': session.query(AnalysisResult).count(),
                'total_violations': session.query(Violation).count(),
                'processing_logs': session.query(ProcessingLog).count(),
                'active_sessions': session.query(UserSession).filter(
                    UserSession.last_activity >= datetime.utcnow().replace(hour=0, minute=0, second=0)
                ).count()
            }
            
            # Risk level distribution
            risk_distribution = session.query(
                AnalysisResult.overall_risk_level,
                session.query(AnalysisResult).filter(
                    AnalysisResult.overall_risk_level == AnalysisResult.overall_risk_level
                ).count()
            ).group_by(AnalysisResult.overall_risk_level).all()
            
            stats['risk_distribution'] = {level: count for level, count in risk_distribution}
            
            return stats
            
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {}

# Export main classes and functions
__all__ = [
    'Base', 'Document', 'AnalysisResult', 'Violation', 'ProcessingLog',
    'ModelPerformance', 'UserSession', 'SystemMetrics', 'ContentRule',
    'DatabaseManager', 'get_db_session', 'DocumentDAL', 'AnalysisResultDAL',
    'ViolationDAL', 'init_database', 'get_database_stats'
]