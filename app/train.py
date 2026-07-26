"""
Training pipeline for logging model metrics and artifacts to MLflow.
"""

import mlflow
import torch
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from app.model_loader import ModelLoader
from app.inference import InferenceEngine
from app.utils import get_hardware_info, save_json_artifact, Timer, get_model_params
from app.config import settings

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Manages the MLflow experiment tracking and model logging."""
    
    def __init__(self):
        """Initialize the training pipeline."""
        self.experiment_name = settings.MLFLOW_EXPERIMENT_NAME
        self.model_name = settings.MODEL_NAME
        self.max_length = settings.MAX_LENGTH
        self.temperature = settings.TEMPERATURE
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        
        # Set or create experiment
        self.experiment_id = mlflow.set_experiment(self.experiment_name)
        
        logger.info(f"MLflow experiment set: {self.experiment_name}")
        logger.info(f"Tracking URI: {settings.MLFLOW_TRACKING_URI}")
    
    def run_pipeline(self, test_prompts: list = None) -> Dict[str, Any]:
        """
        Run the complete training pipeline with MLflow tracking.
        
        Args:
            test_prompts: List of prompts for testing
        
        Returns:
            Dictionary with pipeline results
        """
        if test_prompts is None:
            test_prompts = [
                "Explain MLOps in simple terms.",
                "What is the future of AI?",
                "Write a short story about a robot."
            ]
        
        results = {}
        
        with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            try:
                # Step 1: Load model
                logger.info("Loading model...")
                loader = ModelLoader(self.model_name, use_gpu=settings.USE_GPU)
                model, tokenizer = loader.load_model()
                
                # Get model info
                model_info = loader.get_model_info()
                total_params = get_model_params(model)
                model_info["total_parameters"] = total_params
                
                # Log model parameters
                mlflow.log_params({
                    "model_name": self.model_name,
                    "max_length": self.max_length,
                    "temperature": self.temperature,
                    "device": model_info.get("device", "cpu"),
                    "total_parameters": total_params,
                    "use_gpu": settings.USE_GPU
                })
                
                # Log model configuration
                model_config = model_info.get("model_config", {})
                mlflow.log_params({
                    f"config_{k}": v for k, v in model_config.items() 
                    if isinstance(v, (str, int, float, bool))
                })
                
                # Step 2: Hardware info
                hardware_info = get_hardware_info()
                mlflow.log_params({
                    "hardware": str(hardware_info)
                })
                
                # Step 3: Run inference tests
                logger.info("Running inference tests...")
                engine = InferenceEngine(model, tokenizer, self.max_length, self.temperature)
                
                all_results = []
                total_time = 0
                
                for idx, prompt in enumerate(test_prompts):
                    logger.info(f"Testing prompt {idx+1}/{len(test_prompts)}")
                    
                    with Timer() as timer:
                        result = engine.generate_text(prompt)
                    
                    all_results.append(result)
                    total_time += timer.duration
                    
                    # Log metrics for each prompt
                    mlflow.log_metrics({
                        f"inference_time_prompt_{idx}": result["inference_time"],
                        f"tokens_second_prompt_{idx}": result["tokens_per_second"],
                        f"total_tokens_prompt_{idx}": result["total_tokens"]
                    })
                    
                    # Save generated text as artifact
                    artifact_filename = f"generated_response_{idx}.json"
                    artifact_path = save_json_artifact(
                        result,
                        artifact_filename,
                        settings.ARTIFACTS_DIR
                    )
                    
                    # Log artifact to MLflow
                    mlflow.log_artifact(str(artifact_path))
                
                # Log aggregate metrics
                avg_inference_time = total_time / len(test_prompts)
                mlflow.log_metrics({
                    "avg_inference_time": avg_inference_time,
                    "total_test_prompts": len(test_prompts),
                    "total_inference_time": total_time
                })
                
                # Step 4: Log model to MLflow - FIXED
                logger.info("Logging model to MLflow...")
                
                # Create a simple signature
                from mlflow.models import infer_signature
                
                # Create sample input and output
                sample_prompt = "This is a sample prompt"
                sample_input_ids = tokenizer.encode(sample_prompt, return_tensors="pt")
                
                with torch.no_grad():
                    sample_output = model.generate(
                        sample_input_ids.to(model.device),
                        max_length=50,
                        temperature=0.7,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=True,
                        top_p=0.95,
                        top_k=50
                    )
                sample_output_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)
                
                # Infer signature
                signature = infer_signature(
                    sample_prompt, 
                    {"generated_text": sample_output_text}
                )
                
                # Log model with explicit pip requirements to avoid torchvision detection
                try:
                    # First try with transformers flavor
                    mlflow.transformers.log_model(
                        transformers_model={"model": model, "tokenizer": tokenizer},
                        artifact_path="distilgpt2_model",
                        signature=signature,
                        registered_model_name=settings.MODEL_REGISTRY_NAME,
                        input_example="This is a test prompt",
                        pip_requirements=[
                            "torch==2.6.0",
                            "transformers==4.36.2",
                            "accelerate==0.25.0"
                        ]
                    )
                except Exception as e:
                    logger.warning(f"Transformers flavor logging failed: {e}")
                    logger.info("Trying alternative logging method...")
                    
                    # Alternative: Use pyfunc model
                    from mlflow.models import Model
                    from mlflow.pyfunc import PythonModel, PythonModelContext
                    from mlflow.models.signature import ModelSignature
                    from mlflow.types import Schema, ColSpec, DataType
                    
                    class DistilGPT2Wrapper(PythonModel):
                        def __init__(self, model, tokenizer):
                            self.model = model
                            self.tokenizer = tokenizer
                        
                        def predict(self, context, model_input):
                            import torch
                            prompt = model_input["prompt"].values[0]
                            inputs = self.tokenizer(
                                prompt,
                                return_tensors="pt",
                                truncation=True,
                                max_length=100
                            )
                            with torch.no_grad():
                                outputs = self.model.generate(
                                    **inputs,
                                    max_length=100,
                                    temperature=0.8,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    do_sample=True,
                                    top_p=0.95,
                                    top_k=50
                                )
                            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Create wrapper
                    wrapper = DistilGPT2Wrapper(model, tokenizer)
                    
                    # Define signature
                    signature = ModelSignature(
                        inputs=Schema([ColSpec(DataType.string, "prompt")]),
                        outputs=Schema([ColSpec(DataType.string, "generated_text")])
                    )
                    
                    # Log as PyFunc
                    mlflow.pyfunc.log_model(
                        artifact_path="distilgpt2_model",
                        python_model=wrapper,
                        signature=signature,
                        registered_model_name=settings.MODEL_REGISTRY_NAME,
                        pip_requirements=[
                            "torch==2.6.0",
                            "transformers==4.36.2",
                            "accelerate==0.25.0"
                        ]
                    )
                
                # Step 5: Register model
                logger.info("Registering model...")
                model_version = self.register_model(run.info.run_id)
                
                # Prepare results
                results = {
                    "status": "success",
                    "run_id": run.info.run_id,
                    "experiment_id": self.experiment_id,
                    "model_registry_name": settings.MODEL_REGISTRY_NAME,
                    "model_version": model_version,
                    "test_results": all_results,
                    "total_parameters": total_params,
                    "average_inference_time": avg_inference_time
                }
                
                logger.info("Pipeline completed successfully!")
                return results
                
            except Exception as e:
                logger.error(f"Pipeline failed: {str(e)}")
                mlflow.log_param("error", str(e))
                raise
    
    def register_model(self, run_id: str) -> int:
        """
        Register the model in MLflow Model Registry.
        
        Args:
            run_id: MLflow run ID
        
        Returns:
            Model version number
        """
        try:
            # Get the model URI
            model_uri = f"runs:/{run_id}/distilgpt2_model"
            
            # Create model version
            model_version = mlflow.register_model(
                model_uri,
                settings.MODEL_REGISTRY_NAME
            )
            
            logger.info(f"Model registered with version: {model_version.version}")
            return model_version.version
            
        except Exception as e:
            logger.error(f"Failed to register model: {str(e)}")
            raise


def main():
    """Main entry point for training pipeline."""
    pipeline = TrainingPipeline()
    results = pipeline.run_pipeline()
    print("\n" + "=" * 60)
    print("Pipeline Results:")
    print("=" * 60)
    print(f"Status: {results['status']}")
    print(f"Run ID: {results['run_id']}")
    print(f"Model Registry: {results['model_registry_name']}")
    print(f"Model Version: {results['model_version']}")
    print(f"Total Parameters: {results['total_parameters']:,}")
    print(f"Average Inference Time: {results['average_inference_time']:.3f}s")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()