"""
Configuration management for the MLflow project.
Handles environment variables and application settings.
"""

import os
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Settings:
    """Application configuration settings."""
    
    # MLflow settings
    MLFLOW_TRACKING_URI: str = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    MLFLOW_EXPERIMENT_NAME: str = field(
        default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT_NAME", "DistilGPT2-MLOps")
    )
    
    # Model settings
    MODEL_NAME: str = field(
        default_factory=lambda: os.getenv("MODEL_NAME", "distilgpt2")
    )
    MODEL_REGISTRY_NAME: str = field(
        default_factory=lambda: os.getenv("MODEL_REGISTRY_NAME", "distilgpt2_model")
    )
    MAX_LENGTH: int = field(
        default_factory=lambda: int(os.getenv("MAX_LENGTH", "100"))
    )
    TEMPERATURE: float = field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", "0.8"))
    )
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"
    LOGS_DIR: Path = BASE_DIR / "logs"
    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
    
    # API settings
    API_HOST: str = field(
        default_factory=lambda: os.getenv("API_HOST", "0.0.0.0")
    )
    API_PORT: int = field(
        default_factory=lambda: int(os.getenv("API_PORT", "8000"))
    )
    
    # Hardware settings
    USE_GPU: bool = field(
        default_factory=lambda: os.getenv("USE_GPU", "false").lower() == "true"
    )
    
    def __post_init__(self):
        """Create necessary directories after initialization."""
        for directory in [self.MODELS_DIR, self.LOGS_DIR, self.ARTIFACTS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Create Settings instance from environment variables."""
        return cls()


# Global settings instance
settings = Settings()