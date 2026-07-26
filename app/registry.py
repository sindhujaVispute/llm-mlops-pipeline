"""
Model registry utilities for loading registered models from MLflow.
"""

import mlflow
import logging
from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.config import settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Handles loading models from MLflow Model Registry."""
    
    def __init__(self):
        """Initialize the model registry."""
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        self.registry_name = settings.MODEL_REGISTRY_NAME
    
    def load_latest_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load the latest version of the model from MLflow registry.
        
        Returns:
            Tuple of (model, tokenizer)
        
        Raises:
            Exception: If model loading fails
        """
        try:
            logger.info(f"Loading latest model from registry: {self.registry_name}")
            
            # Get the latest version
            model_uri = f"models:/{self.registry_name}/latest"
            
            # Load the model using transformers flavor
            loaded_model = mlflow.transformers.load_model(model_uri)
            
            model = loaded_model.get("model")
            tokenizer = loaded_model.get("tokenizer")
            
            if model is None or tokenizer is None:
                raise ValueError("Loaded model does not contain model or tokenizer")
            
            logger.info(f"Model loaded successfully from registry")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model from registry: {str(e)}")
            raise
    
    def load_version(self, version: int) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load a specific version of the model from MLflow registry.
        
        Args:
            version: Model version number
        
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            logger.info(f"Loading model version {version} from registry: {self.registry_name}")
            
            model_uri = f"models:/{self.registry_name}/{version}"
            loaded_model = mlflow.transformers.load_model(model_uri)
            
            model = loaded_model.get("model")
            tokenizer = loaded_model.get("tokenizer")
            
            if model is None or tokenizer is None:
                raise ValueError("Loaded model does not contain model or tokenizer")
            
            logger.info(f"Model version {version} loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model version {version}: {str(e)}")
            raise
    
    def list_versions(self) -> list:
        """
        List all versions of the model in the registry.
        
        Returns:
            List of model versions
        """
        try:
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(self.registry_name)
            
            return [{
                "version": v.version,
                "status": v.status,
                "stage": v.stage,
                "run_id": v.run_id,
                "creation_timestamp": v.creation_timestamp
            } for v in versions]
            
        except Exception as e:
            logger.error(f"Failed to list model versions: {str(e)}")
            raise