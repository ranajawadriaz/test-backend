"""
MLflow integration for tracking experiments and logging models
"""
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import os
from datetime import datetime

class MLflowTracker:
    """Wrapper for MLflow experiment tracking"""
    
    def __init__(self, experiment_name="audio-deepfake-detection"):
        """Initialize MLflow tracking"""
        # Set tracking URI from environment or default
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"MLflow setup warning: {e}")
    
    def log_prediction(self, filename, result, duration):
        """Log individual prediction results"""
        try:
            with mlflow.start_run(run_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters
                mlflow.log_param("filename", filename)
                mlflow.log_param("num_chunks", result["num_chunks"])
                mlflow.log_param("audio_duration", result["duration_seconds"])
                mlflow.log_param("processing_time", duration)
                
                # Log metrics
                mlflow.log_metric("ensemble_confidence", result["confidence"])
                mlflow.log_metric("rf_confidence", result["individual_models"]["random_forest"]["confidence"])
                mlflow.log_metric("cnn_confidence", result["individual_models"]["cnn"]["confidence"])
                
                # Log predictions
                mlflow.log_param("ensemble_prediction", result["prediction"])
                mlflow.log_param("rf_prediction", result["individual_models"]["random_forest"]["prediction"])
                mlflow.log_param("cnn_prediction", result["individual_models"]["cnn"]["prediction"])
                
                # Log probabilities
                mlflow.log_metric("ensemble_bonafide_prob", result["probabilities"]["bonafide"])
                mlflow.log_metric("ensemble_spoof_prob", result["probabilities"]["spoof"])
                
                # Log tags
                mlflow.set_tag("prediction_type", result["prediction"])
                mlflow.set_tag("model_type", "ensemble")
                
        except Exception as e:
            print(f"MLflow logging warning: {e}")
    
    def log_batch_results(self, results, total_time):
        """Log batch prediction results"""
        try:
            with mlflow.start_run(run_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log batch metrics
                mlflow.log_param("total_files", len(results))
                mlflow.log_param("batch_processing_time", total_time)
                
                bonafide_count = sum(1 for r in results if r.get("prediction") == "BONAFIDE")
                spoof_count = len(results) - bonafide_count
                
                mlflow.log_metric("bonafide_count", bonafide_count)
                mlflow.log_metric("spoof_count", spoof_count)
                mlflow.log_metric("avg_time_per_file", total_time / len(results) if results else 0)
                
                # Log average confidences
                avg_confidence = sum(r.get("confidence", 0) for r in results) / len(results) if results else 0
                mlflow.log_metric("avg_confidence", avg_confidence)
                
                mlflow.set_tag("batch_size", len(results))
                
        except Exception as e:
            print(f"MLflow batch logging warning: {e}")
    
    def log_model_info(self, model_name, model_path, metrics=None):
        """Log model metadata and metrics"""
        try:
            with mlflow.start_run(run_name=f"model_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("model_path", model_path)
                
                if metrics:
                    for key, value in metrics.items():
                        mlflow.log_metric(key, value)
                
                mlflow.set_tag("model_type", model_name)
                
        except Exception as e:
            print(f"MLflow model logging warning: {e}")
