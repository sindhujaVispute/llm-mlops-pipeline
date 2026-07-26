"""
Inference engine for text generation using the loaded model.
"""

import torch
import time
import logging
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Handles text generation using the loaded model."""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                 max_length: int = 100, temperature: float = 0.8):
        """
        Initialize the inference engine.
        
        Args:
            model: Loaded Hugging Face model
            tokenizer: Loaded tokenizer
            max_length: Maximum length of generated text
            temperature: Temperature for sampling
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.temperature = temperature
        self.device = next(model.parameters()).device
    
    def generate_text(self, prompt: str) -> Dict[str, Any]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
        
        Returns:
            Dictionary containing generated text and metadata
        """
        try:
            logger.info(f"Generating text for prompt: {prompt[:50]}...")
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            # Measure inference time
            start_time = time.time()
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50
                )
            
            inference_time = time.time() - start_time
            
            # Decode generated text
            generated_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # Calculate tokens per second
            total_tokens = outputs.shape[1]
            tokens_per_second = total_tokens / inference_time if inference_time > 0 else 0
            
            result = {
                "prompt": prompt,
                "generated_text": generated_text,
                "inference_time": inference_time,
                "total_tokens": total_tokens,
                "tokens_per_second": tokens_per_second,
                "parameters": {
                    "max_length": self.max_length,
                    "temperature": self.temperature
                }
            }
            
            logger.info(f"Generation completed in {inference_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            raise
    
    def __call__(self, prompt: str) -> Dict[str, Any]:
        """Make the engine callable."""
        return self.generate_text(prompt)