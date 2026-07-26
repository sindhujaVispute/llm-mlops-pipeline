"""
FastAPI application for serving the DistilGPT-2 model with MLflow integration.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uvicorn
import logging
import sys
import os
from contextlib import asynccontextmanager
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.utils import setup_logging, get_hardware_info
from app.model_loader import ModelLoader
from app.inference import InferenceEngine
from app.registry import ModelRegistry
from app.train import TrainingPipeline

# Setup logging
logger = setup_logging()


# Pydantic models for request/response
class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="Input text prompt for generation", example="Explain MLOps")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Explain MLOps in simple terms"
            }
        }


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    response: str = Field(..., description="Generated text response")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata about the generation")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    hardware: Optional[Dict[str, Any]] = Field(None, description="Hardware information")


class TrainResponse(BaseModel):
    """Training pipeline response."""
    status: str = Field(..., description="Training status")
    results: Optional[Dict[str, Any]] = Field(None, description="Training results")
    message: Optional[str] = Field(None, description="Additional message")


class VersionResponse(BaseModel):
    """Model versions response."""
    versions: list = Field(..., description="List of model versions")


# Global variables
engine = None
model_info = None
loader = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    global engine, model_info, loader
    
    # Startup
    logger.info("=" * 60)
    logger.info("Starting up DistilGPT-2 MLOps API Service...")
    logger.info("=" * 60)
    
    try:
        # Get hardware info
        hw_info = get_hardware_info()
        logger.info(f"Hardware: {hw_info['cpu']['cores']} CPU cores, {hw_info['memory']['total'] / (1024**3):.1f} GB RAM")
        logger.info(f"GPU available: {hw_info['gpu']['available']}")
        
        # Try to load from registry first
        logger.info("Attempting to load model from MLflow registry...")
        registry = ModelRegistry()
        model, tokenizer = registry.load_latest_model()
        
        # Initialize inference engine
        engine = InferenceEngine(
            model=model,
            tokenizer=tokenizer,
            max_length=settings.MAX_LENGTH,
            temperature=settings.TEMPERATURE
        )
        
        model_info = {
            "model_name": settings.MODEL_NAME,
            "registry_name": settings.MODEL_REGISTRY_NAME,
            "device": str(engine.device),
            "max_length": settings.MAX_LENGTH,
            "temperature": settings.TEMPERATURE,
            "model_loaded_from": "registry"
        }
        
        logger.info(f"✅ Model loaded from registry successfully on {engine.device}")
        
    except Exception as e:
        logger.warning(f"Could not load model from registry: {e}")
        logger.info("Loading model directly from Hugging Face...")
        
        try:
            # Fallback to direct loading
            loader = ModelLoader(settings.MODEL_NAME, use_gpu=settings.USE_GPU)
            model, tokenizer = loader.load_model()
            
            engine = InferenceEngine(
                model=model,
                tokenizer=tokenizer,
                max_length=settings.MAX_LENGTH,
                temperature=settings.TEMPERATURE
            )
            
            model_info = loader.get_model_info()
            model_info["max_length"] = settings.MAX_LENGTH
            model_info["temperature"] = settings.TEMPERATURE
            model_info["model_loaded_from"] = "huggingface"
            
            logger.info(f"✅ Model loaded from Hugging Face successfully on {engine.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            engine = None
            model_info = {"error": str(e)}
    
    # Log startup info
    logger.info(f"API will run on http://{settings.API_HOST}:{settings.API_PORT}")
    logger.info(f"MLflow tracking URI: {settings.MLFLOW_TRACKING_URI}")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("=" * 60)
    logger.info("Shutting down FastAPI application...")
    logger.info("=" * 60)


# Create FastAPI app
app = FastAPI(
    title="DistilGPT-2 MLOps API",
    description="""
    ## 🤖 Text Generation API with MLflow
    
    This API serves a DistilGPT-2 language model with full MLflow integration.
    
    ### Features:
    - Text generation with configurable parameters
    - MLflow experiment tracking
    - Model registry integration
    - Automatic model loading from registry
    - Hardware detection and logging
    - Comprehensive error handling
    
    ### Endpoints:
    - `GET /` - Health check
    - `POST /generate` - Generate text from prompt
    - `GET /train` - Trigger training pipeline
    - `GET /versions` - List model versions
    - `GET /info` - Get model and system information
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.get("/", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status of the service with model and hardware information
    """
    hw_info = get_hardware_info()
    
    return HealthResponse(
        status="healthy" if engine is not None else "degraded",
        version="1.0.0",
        model_loaded=engine is not None,
        hardware={
            "cpu_cores": hw_info["cpu"]["cores"],
            "memory_gb": hw_info["memory"]["total"] / (1024**3),
            "gpu_available": hw_info["gpu"]["available"],
            "gpu_name": hw_info["gpu"].get("name", "None") if hw_info["gpu"]["available"] else "None"
        }
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """
    Generate text from a prompt.
    
    Args:
        request: GenerateRequest with prompt
        
    Returns:
        Generated text response with metadata
        
    Raises:
        HTTPException: If generation fails or model not loaded
    """
    if not engine:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable. Please check logs."
        )
    
    try:
        logger.info(f"📝 Received generation request: {request.prompt[:50]}...")
        
        # Generate text
        result = engine.generate_text(request.prompt)
        
        # Prepare response
        response = GenerateResponse(
            response=result["generated_text"],
            metadata={
                "inference_time": result["inference_time"],
                "total_tokens": result["total_tokens"],
                "tokens_per_second": result["tokens_per_second"],
                "model_info": model_info,
                "generation_params": {
                    "max_length": settings.MAX_LENGTH,
                    "temperature": settings.TEMPERATURE
                }
            }
        )
        
        logger.info(f"✅ Generation completed in {result['inference_time']:.3f}s ({result['tokens_per_second']:.1f} tokens/sec)")
        return response
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Text generation failed: {str(e)}"
        )


@app.get("/train", response_model=TrainResponse)
async def trigger_training():
    """
    Trigger the training pipeline to update the model.
    
    This will:
    1. Load the model
    2. Run inference tests
    3. Log metrics to MLflow
    4. Register the model
    5. Reload the latest model
    
    Returns:
        Training pipeline results
    """
    global engine, model_info
    
    try:
        logger.info("🔄 Triggering training pipeline...")
        
        # Run training pipeline
        pipeline = TrainingPipeline()
        results = pipeline.run_pipeline()
        
        # Reload model after training
        logger.info("Reloading model from registry...")
        registry = ModelRegistry()
        model, tokenizer = registry.load_latest_model()
        
        engine = InferenceEngine(
            model=model,
            tokenizer=tokenizer,
            max_length=settings.MAX_LENGTH,
            temperature=settings.TEMPERATURE
        )
        
        model_info = {
            "model_name": settings.MODEL_NAME,
            "registry_name": settings.MODEL_REGISTRY_NAME,
            "device": str(engine.device),
            "max_length": settings.MAX_LENGTH,
            "temperature": settings.TEMPERATURE,
            "model_loaded_from": "registry"
        }
        
        logger.info("✅ Training completed and model reloaded successfully")
        
        return TrainResponse(
            status="success",
            results=results,
            message="Model trained and registered successfully"
        )
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Training pipeline failed: {str(e)}"
        )


@app.get("/versions", response_model=VersionResponse)
async def get_model_versions():
    """
    Get all versions of the model in the MLflow registry.
    
    Returns:
        List of model versions with their status and stage
    """
    try:
        registry = ModelRegistry()
        versions = registry.list_versions()
        
        logger.info(f"Retrieved {len(versions)} model versions")
        return VersionResponse(versions=versions)
        
    except Exception as e:
        logger.error(f"❌ Failed to get model versions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model versions: {str(e)}"
        )


@app.get("/info")
async def get_info():
    """
    Get detailed model and system information.
    
    Returns:
        Comprehensive information about the model, hardware, and configuration
    """
    hw_info = get_hardware_info()
    
    return {
        "model": model_info,
        "hardware": hw_info,
        "configuration": {
            "mlflow_tracking_uri": settings.MLFLOW_TRACKING_URI,
            "experiment_name": settings.MLFLOW_EXPERIMENT_NAME,
            "model_name": settings.MODEL_NAME,
            "registry_name": settings.MODEL_REGISTRY_NAME,
            "max_length": settings.MAX_LENGTH,
            "temperature": settings.TEMPERATURE,
            "use_gpu": settings.USE_GPU
        }
    }


@app.get("/health")
async def health():
    """
    Simple health check endpoint.
    
    Returns:
        Basic health status
    """
    return {
        "status": "healthy" if engine is not None else "degraded",
        "model_loaded": engine is not None
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    Custom HTTP exception handler.
    """
    logger.error(f"HTTP Exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    General exception handler for unhandled exceptions.
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
        log_level="info",
        access_log=True
    )