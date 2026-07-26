"""
Utility functions for logging, hardware detection, and common operations.
"""

import logging
import sys
import platform
import torch
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
import json
import psutil


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log')
        ]
    )
    return logging.getLogger(__name__)


def get_hardware_info() -> Dict[str, Any]:
    """
    Get system hardware information.
    
    Returns:
        Dictionary containing hardware information
    """
    info = {
        "cpu": {
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "usage": psutil.cpu_percent(interval=1),
            "architecture": platform.machine(),
            "processor": platform.processor()
        },
        "memory": {
            "total": psutil.virtual_memory().total,
            "available": psutil.virtual_memory().available,
            "used": psutil.virtual_memory().used
        },
        "gpu": {}
    }
    
    # Check for GPU
    if torch.cuda.is_available():
        info["gpu"] = {
            "available": True,
            "count": torch.cuda.device_count(),
            "current": torch.cuda.current_device(),
            "name": torch.cuda.get_device_name(0),
            "memory_allocated": torch.cuda.memory_allocated(0),
            "memory_reserved": torch.cuda.memory_reserved(0)
        }
    else:
        info["gpu"] = {"available": False}
    
    return info


def save_json_artifact(data: Dict[str, Any], filename: str, directory: Path) -> Path:
    """
    Save data as JSON artifact.
    
    Args:
        data: Data to save
        filename: Name of the file
        directory: Directory to save the file
    
    Returns:
        Path to the saved file
    """
    filepath = directory / filename
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    return filepath


def get_model_params(model) -> int:
    """
    Count total parameters in a model.
    
    Args:
        model: PyTorch model instance
    
    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


class Timer:
    """Context manager for timing code execution."""
    
    def __enter__(self):
        self.start = datetime.now()
        return self
    
    def __exit__(self, *args):
        self.end = datetime.now()
        self.duration = (self.end - self.start).total_seconds()