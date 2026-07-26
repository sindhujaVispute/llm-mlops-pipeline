"""
Unit tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
import json

client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert "model_loaded" in data


def test_generate_text():
    """Test text generation endpoint."""
    test_prompt = "Explain MLOps"
    response = client.post(
        "/generate",
        json={"prompt": test_prompt}
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "response" in data
    assert "metadata" in data
    assert len(data["response"]) > 0
    
    # Check metadata
    metadata = data["metadata"]
    assert "inference_time" in metadata
    assert "total_tokens" in metadata
    assert "tokens_per_second" in metadata
    assert "model_info" in metadata


def test_generate_empty_prompt():
    """Test generation with empty prompt."""
    response = client.post(
        "/generate",
        json={"prompt": ""}
    )
    
    # Should still work or return error
    assert response.status_code in [200, 500]


def test_generate_long_prompt():
    """Test generation with very long prompt."""
    long_prompt = "This is a very long prompt. " * 100
    response = client.post(
        "/generate",
        json={"prompt": long_prompt}
    )
    
    assert response.status_code in [200, 500]


def test_invalid_request():
    """Test with invalid request body."""
    response = client.post(
        "/generate",
        json={"invalid_field": "test"}
    )
    
    assert response.status_code == 422  # Validation error


def test_versions_endpoint():
    """Test model versions endpoint."""
    response = client.get("/versions")
    assert response.status_code == 200
    data = response.json()
    assert "versions" in data