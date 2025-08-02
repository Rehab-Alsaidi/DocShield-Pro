# utils/logger.py
import logging
import logging.handlers
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)

class ContextFilter(logging.Filter):
    """Add contextual information to log records"""
    
    def filter(self, record):
        # Add process ID
        record.pid = os.getpid()
        
        # Add module name (shortened)
        if hasattr(record, 'module'):
            parts = record.module.split('.')
            record.short_module = parts[-1] if parts else record.module
        
        return True

def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = True,
    colored_output: bool = True
) -> None:
    """Setup comprehensive logging configuration"""
    
    # Create logs directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(pid)d | %(name)-20s | %(levelname)-8s | %(funcName)-15s:%(lineno)-3d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Create colored formatter for console
    colored_formatter = ColoredFormatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Create context filter
    context_filter = ContextFilter()
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        if colored_output and sys.stdout.isatty():
            console_handler.setFormatter(colored_formatter)
        else:
            console_handler.setFormatter(simple_formatter)
        
        console_handler.addFilter(context_filter)
        root_logger.addHandler(console_handler)
    
    # File handler (rotating)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        file_handler.addFilter(context_filter)
        root_logger.addHandler(file_handler)
    
    # Error file handler (separate file for errors)
    if log_file:
        error_file = log_file.replace('.log', '_errors.log')
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        error_handler.addFilter(context_filter)
        root_logger.addHandler(error_handler)

def get_logger(name: str) -> logging.Logger:
    """Get logger instance with specified name"""
    return logging.getLogger(name)

def log_function_call(func):
    """Decorator to log function calls"""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={len(args)}, kwargs={list(kwargs.keys())}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Function {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Function {func.__name__} failed: {e}")
            raise
    
    return wrapper

def log_execution_time(func):
    """Decorator to log function execution time"""
    def wrapper(*args, **kwargs):
        import time
        logger = get_logger(func.__module__)
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Function {func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Function {func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    return wrapper

class LoggingContext:
    """Context manager for temporary logging configuration"""
    
    def __init__(self, logger_name: str, level: int = None, extra_context: dict = None):
        self.logger = get_logger(logger_name)
        self.original_level = self.logger.level
        self.new_level = level
        self.extra_context = extra_context or {}
    
    def __enter__(self):
        if self.new_level is not None:
            self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)

def setup_application_logging():
    """Setup standard application logging configuration"""
    
    # Determine log level from environment
    log_level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    # Create logs directory
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Setup main logging
    setup_logging(
        log_level=log_level,
        log_file='logs/app.log',
        console_output=True,
        colored_output=True
    )
    
    # Configure specific loggers
    configure_third_party_loggers()
    
    # Log startup message
    logger = get_logger(__name__)
    logger.info("Application logging configured successfully")
    logger.info(f"Log level: {logging.getLevelName(log_level)}")

def configure_third_party_loggers():
    """Configure logging for third-party libraries"""
    
    # Suppress noisy loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # Flask logging
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
    # SQLAlchemy logging (if used)
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)

def log_system_info():
    """Log system information"""
    import platform
    import psutil
    
    logger = get_logger(__name__)
    
    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python Version: {platform.python_version()}")
    logger.info(f"CPU Count: {psutil.cpu_count()}")
    logger.info(f"Memory: {psutil.virtual_memory().total // (1024**3)} GB")
    
    # GPU information (if available)
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA Available: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB")
        else:
            logger.info("CUDA: Not available")
    except ImportError:
        logger.info("PyTorch: Not installed")
    
    logger.info("=== End System Information ===")

class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self, name: str):
        self.logger = get_logger(f"performance.{name}")
        self.metrics = {}
    
    def start_timer(self, metric_name: str):
        """Start timing a metric"""
        import time
        self.metrics[metric_name] = {'start': time.time()}
    
    def end_timer(self, metric_name: str):
        """End timing a metric and log result"""
        import time
        if metric_name in self.metrics:
            duration = time.time() - self.metrics[metric_name]['start']
            self.logger.info(f"{metric_name}: {duration:.3f}s")
            self.metrics[metric_name]['duration'] = duration
            return duration
        return None
    
    def log_metric(self, metric_name: str, value: float, unit: str = ""):
        """Log a custom metric"""
        self.logger.info(f"{metric_name}: {value}{unit}")
    
    def get_summary(self) -> dict:
        """Get summary of all metrics"""
        return {
            name: data.get('duration', 0) 
            for name, data in self.metrics.items()
        }

# Error logging utilities
def log_exception(logger: logging.Logger, message: str = "Exception occurred"):
    """Log exception with full traceback"""
    import traceback
    logger.error(f"{message}: {traceback.format_exc()}")

def log_model_loading(model_name: str, success: bool, load_time: float = None):
    """Log model loading results"""
    logger = get_logger("model_loading")
    
    if success:
        time_str = f" in {load_time:.2f}s" if load_time else ""
        logger.info(f"Successfully loaded model: {model_name}{time_str}")
    else:
        logger.error(f"Failed to load model: {model_name}")

def log_processing_stats(file_name: str, stats: dict):
    """Log processing statistics"""
    logger = get_logger("processing_stats")
    
    logger.info(f"Processed: {file_name}")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

# Initialize logging when module is imported
if not logging.getLogger().handlers:
    setup_application_logging()