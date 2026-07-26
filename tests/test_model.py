"""
Unit tests for model loading and inference.
"""

import pytest
import torch
from app.model_loader import ModelLoader
from app.inference import InferenceEngine
from app.config import settings
from app.utils import get_model_params, get_hardware_info


def test_model_loader():
    """Test model loader functionality."""
    loader = ModelLoader(settings.MODEL_NAME, use_gpu=False)
    model, tokenizer = loader.load_model()
    
    assert model is not None
    assert tokenizer is not None
    assert hasattr(model, "generate")
    assert hasattr(tokenizer, "encode")


def test_model_info():
    """Test model info retrieval."""
    loader = ModelLoader(settings.MODEL_NAME, use_gpu=False)
    model, _ = loader.load_model()
    info = loader.get_model_info()
    
    assert "model_name" in info
    assert "device" in info
    assert "total_parameters" in info
    assert info["total_parameters"] > 0


def test_inference_engine():
    """Test inference engine functionality."""
    loader = ModelLoader(settings.MODEL_NAME, use_gpu=False)
    model, tokenizer = loader.load_model()
    
    engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        max_length=50,
        temperature=0.7
    )
    
    test_prompt = "Hello, world!"
    result = engine.generate_text(test_prompt)
    
    assert "prompt" in result
    assert "generated_text" in result
    assert "inference_time" in result
    assert "total_tokens" in result
    assert "tokens_per_second" in result
    assert len(result["generated_text"]) > 0


def test_tokenizer_padding():
    """Test tokenizer padding token setup."""
    loader = ModelLoader(settings.MODEL_NAME, use_gpu=False)
    _, tokenizer = loader.load_model()
    
    assert tokenizer.pad_token is not None
    assert tokenizer.pad_token == tokenizer.eos_token


def test_device_assignment():
    """Test proper device assignment."""
    loader = ModelLoader(settings.MODEL_NAME, use_gpu=False)
    model, _ = loader.load_model()
    
    device = next(model.parameters()).device
    assert device.type in ["cpu", "cuda"]


def test_parameters_count():
    """Test parameter counting."""
    loader = ModelLoader(settings.MODEL_NAME, use_gpu=False)
    model, _ = loader.load_model()
    
    params = get_model_params(model)
    assert params > 0
    assert isinstance(params, int)


def test_hardware_info():
    """Test hardware info retrieval."""
    info = get_hardware_info()
    
    assert "cpu" in info
    assert "memory" in info
    assert "gpu" in info
    assert "cores" in info["cpu"]
    assert "total" in info["memory"]


# Removed @pytest.mark.slow since it wasn't registered
def test_batch_generation():
    """Test generating multiple texts."""
    loader = ModelLoader(settings.MODEL_NAME, use_gpu=False)
    model, tokenizer = loader.load_model()
    
    engine = InferenceEngine(model, tokenizer, max_length=50, temperature=0.7)
    
    prompts = [
        "Explain AI",
        "What is machine learning?",
        "Define deep learning"
    ]
    
    results = []
    for prompt in prompts:
        result = engine.generate_text(prompt)
        results.append(result)
    
    assert len(results) == len(prompts)
    for result in results:
        assert "generated_text" in result
        assert len(result["generated_text"]) > 0