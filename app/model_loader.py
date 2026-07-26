"""
Model loading utilities for Hugging Face transformers.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading of Hugging Face models and tokenizers."""
    
    def __init__(self, model_name: str, use_gpu: bool = False):
        """
        Initialize the ModelLoader.
        
        Args:
            model_name: Name of the model to load
            use_gpu: Whether to use GPU for inference
        """
        self.model_name = model_name
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
    
    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load the model and tokenizer from Hugging Face.
        
        Returns:
            Tuple of (model, tokenizer)
        
        Raises:
            Exception: If model loading fails
        """
        try:
            logger.info(f"Loading model: {self.model_name} on {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir="models/"
            )
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir="models/"
            )
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self.model:
            return {"error": "Model not loaded"}
        
        return {
            "model_name": self.model_name,
            "device": self.device,
            "model_config": self.model.config.to_dict(),
            "total_parameters": sum(p.numel() for p in self.model.parameters())
        }