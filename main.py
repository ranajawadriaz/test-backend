"""
FastAPI Backend for Audio Deepfake Detection
Provides REST API endpoints for uploading audio files and getting predictions
Includes user authentication with JWT tokens (2-hour expiry)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import tempfile
import os
from pathlib import Path
from typing import Dict, Any
import logging
import time

from run_prediction import AudioDeepfakeDetector
from mlflow_tracker import MLflowTracker
from database import init_db
from auth import router as auth_router, get_current_user
from models import User

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for detector and MLflow tracker
detector = None
mlflow_tracker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources on startup/shutdown"""
    global detector, mlflow_tracker
    
    # Startup
    try:
        # Initialize database and create tables
        logger.info("Initializing database...")
        init_db()
        logger.info("Database initialized successfully!")
        
        logger.info("Loading models...")
        detector = AudioDeepfakeDetector(params_file='params.yaml')
        logger.info("Models loaded successfully!")
        
        logger.info("Initializing MLflow tracking...")
        try:
            mlflow_tracker = MLflowTracker()
            logger.info("MLflow tracking initialized!")
        except Exception as mlflow_error:
            logger.warning(f"MLflow initialization failed: {mlflow_error}")
            logger.warning("Continuing without MLflow tracking...")
            mlflow_tracker = None
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield  # Application runs here
    
    # Shutdown (cleanup if needed)
    logger.info("Shutting down application...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Audio Deepfake Detection API",
    description="Detect AI-generated deepfake audio using ensemble ML models with user authentication",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS for Next.js frontend (local and production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",           # Local development
        "http://localhost:3007",           # Local production port
        "http://143.244.150.120:3000",     # Digital Ocean frontend (port 3000)
        "http://143.244.150.120:3007",     # Digital Ocean frontend (port 3007)
        "http://172.31.23.130:3000",       # Internal IP (port 3000)
        "http://172.31.23.130:3007",       # Internal IP (port 3007)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include authentication routes
app.include_router(auth_router)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Audio Deepfake Detection API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": detector is not None,
        "device": str(detector.device) if detector else "unknown"
    }


@app.post("/predict")
async def predict_audio(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Upload an audio file and get deepfake detection predictions
    **Requires authentication** - Include Bearer token in Authorization header
    
    Args:
        file: Audio file (supports .wav, .mp3, .flac, .ogg, etc.)
        current_user: Authenticated user (automatically injected)
    
    Returns:
        JSON with predictions from all three models and ensemble result
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    logger.info(f"Prediction request from user: {current_user.username}")
    
    # Validate file type
    allowed_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.opus'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file to temporary location
    temp_file = None
    try:
        # Create temp file with original extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp:
            temp_file = temp.name
            content = await file.read()
            temp.write(content)
        
        logger.info(f"Processing file: {file.filename} ({len(content)} bytes)")
        
        # Run prediction with timing
        start_time = time.time()
        result = detector.predict_single(temp_file, verbose=False)
        processing_time = time.time() - start_time
        
        # Format response
        response = {
            "filename": file.filename,
            "file_size_bytes": len(content),
            "duration_seconds": result["duration_seconds"],
            "num_chunks": result["num_chunks"],
            "predictions": {
                "random_forest": {
                    "prediction": result["individual_models"]["random_forest"]["prediction"],
                    "confidence": round(result["individual_models"]["random_forest"]["confidence"] * 100, 2),
                    "probabilities": {
                        "spoof": round(result["individual_models"]["random_forest"]["probabilities"]["spoof"], 4),
                        "bonafide": round(result["individual_models"]["random_forest"]["probabilities"]["bonafide"], 4)
                    }
                },
                "cnn": {
                    "prediction": result["individual_models"]["cnn"]["prediction"],
                    "confidence": round(result["individual_models"]["cnn"]["confidence"] * 100, 2),
                    "probabilities": {
                        "spoof": round(result["individual_models"]["cnn"]["probabilities"]["spoof"], 4),
                        "bonafide": round(result["individual_models"]["cnn"]["probabilities"]["bonafide"], 4)
                    }
                },
                "ensemble": {
                    "prediction": result["prediction"],
                    "confidence": round(result["confidence"] * 100, 2),
                    "probabilities": {
                        "spoof": round(result["probabilities"]["spoof"], 4),
                        "bonafide": round(result["probabilities"]["bonafide"], 4)
                    }
                }
            },
            "final_prediction": result["prediction"],
            "final_confidence": round(result["confidence"] * 100, 2),
            "confidence_level": _get_confidence_level(result["confidence"] * 100),
            "models_agree": (
                result["individual_models"]["random_forest"]["prediction"] == 
                result["individual_models"]["cnn"]["prediction"] == 
                result["prediction"]
            )
        }
        
        logger.info(f"Prediction: {response['final_prediction']} ({response['final_confidence']}%)")
        
        # Log to MLflow
        if mlflow_tracker:
            try:
                mlflow_tracker.log_prediction(file.filename, result, processing_time)
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")


def _get_confidence_level(confidence: float) -> str:
    """Convert confidence percentage to descriptive level"""
    if confidence >= 90:
        return "Very High"
    elif confidence >= 75:
        return "High"
    elif confidence >= 60:
        return "Moderate"
    else:
        return "Low"


@app.post("/batch-predict")
async def batch_predict(
    files: list[UploadFile] = File(...),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Upload multiple audio files for batch prediction
    **Requires authentication** - Include Bearer token in Authorization header
    
    Args:
        files: List of audio files
        current_user: Authenticated user (automatically injected)
    
    Returns:
        JSON with predictions for all files and summary statistics
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    logger.info(f"Batch prediction request from user: {current_user.username} ({len(files)} files)")
    
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files per batch")
    
    results = []
    bonafide_count = 0
    spoof_count = 0
    batch_start_time = time.time()
    
    for file in files:
        try:
            # Validate file type
            allowed_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.opus'}
            file_ext = Path(file.filename).suffix.lower()
            
            if file_ext not in allowed_extensions:
                results.append({
                    "filename": file.filename,
                    "error": f"Unsupported file type: {file_ext}"
                })
                continue
            
            # Process file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp:
                temp_file = temp.name
                content = await file.read()
                temp.write(content)
            
            try:
                result = detector.predict_single(temp_file, verbose=False)
                
                file_result = {
                    "filename": file.filename,
                    "duration": result["duration_seconds"],
                    "prediction": result["prediction"],
                    "confidence": round(result["confidence"] * 100, 2)
                }
                
                results.append(file_result)
                
                if result["prediction"] == "BONAFIDE":
                    bonafide_count += 1
                else:
                    spoof_count += 1
                    
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    batch_total_time = time.time() - batch_start_time
    
    # Log batch results to MLflow
    if mlflow_tracker:
        try:
            mlflow_tracker.log_batch_results(results, batch_total_time)
        except Exception as e:
            logger.warning(f"MLflow batch logging failed: {e}")
    
    return {
        "total_files": len(files),
        "processed": len([r for r in results if "error" not in r]),
        "failed": len([r for r in results if "error" in r]),
        "processing_time": round(batch_total_time, 2),
        "summary": {
            "bonafide": bonafide_count,
            "spoof": spoof_count
        },
        "results": results
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
