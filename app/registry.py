"""
Model registry utilities for loading registered models from MLflow.
"""

import mlflow
import logging
from typing import Tuple, List, Dict, Any
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
            
            # Try to get the latest version
            client = mlflow.tracking.MlflowClient()
            
            # Check if model exists in registry
            try:
                model_versions = client.get_latest_versions(self.registry_name)
                if not model_versions:
                    raise Exception(f"No model versions found for {self.registry_name}")
                
                # Get the latest version
                latest_version = max([int(v.version) for v in model_versions])
                model_uri = f"models:/{self.registry_name}/{latest_version}"
                
            except Exception as e:
                logger.warning(f"Could not get model versions: {e}")
                # Fallback to direct models URI
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
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """
        List all versions of the model in the registry.
        
        Returns:
            List of model versions
        """
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Try to get model versions
            try:
                # Get all versions (not just latest)
                model_versions = client.search_model_versions(f"name='{self.registry_name}'")
                
                if not model_versions:
                    logger.info(f"No versions found for model: {self.registry_name}")
                    return []
                
                versions_list = []
                for v in model_versions:
                    version_data = {
                        "version": v.version,
                        "status": v.status,
                        "run_id": v.run_id,
                        "creation_timestamp": v.creation_timestamp,
                        "last_updated_timestamp": v.last_updated_timestamp
                    }
                    # Check if 'stage' attribute exists (older MLflow versions)
                    if hasattr(v, 'stage'):
                        version_data["stage"] = v.stage
                    else:
                        version_data["stage"] = "None"
                    
                    versions_list.append(version_data)
                
                return versions_list
                
            except Exception as e:
                logger.warning(f"Could not search model versions: {e}")
                
                # Try alternative method
                try:
                    latest_versions = client.get_latest_versions(self.registry_name)
                    versions_list = []
                    for v in latest_versions:
                        version_data = {
                            "version": v.version,
                            "status": v.status,
                            "run_id": v.run_id,
                            "creation_timestamp": v.creation_timestamp
                        }
                        if hasattr(v, 'stage'):
                            version_data["stage"] = v.stage
                        else:
                            version_data["stage"] = "None"
                        versions_list.append(version_data)
                    return versions_list
                except Exception as e2:
                    logger.error(f"Failed to get latest versions: {e2}")
                    return []
            
        except Exception as e:
            logger.error(f"Failed to list model versions: {str(e)}")
            return []