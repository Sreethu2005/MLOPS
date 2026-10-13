import logging
import os

def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a logger with standard formatting.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler (if not running in basic Airflow which handles logs itself,
        # but we add it for local script runs)
        try:
            log_dir = "logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(log_dir, "pipeline.log"))
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except (PermissionError, OSError):
            pass
        
    return logger
