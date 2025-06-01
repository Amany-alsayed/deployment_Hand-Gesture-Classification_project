import logging
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import uvicorn
from app.model import prediction
import os
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Summary
import time
from prometheus_client import Gauge
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(r"app\app.log"),
        logging.StreamHandler()
    ]
)
logger=logging.getLogger(__name__)

try:
    preprocess = joblib.load(os.path.join(MODEL_DIR, "scaling.pkl"))
    logger.info("preprocessing file loaded successfully.")
except Exception as e:
    logger.error(f"failed to load preprosessing file {e}")
    preprocess=None

try:
    model = joblib.load(os.path.join(MODEL_DIR, "SVMClassifier.pkl"))
    logger.info("model loaded successfully.")
except Exception as e:
    logger.error(f"failed to load model {e}")
    model=None


app=FastAPI()

class LandmarkInput(BaseModel):
    landmarks: List[float]

Instrumentator().instrument(app).expose(app)
@app.get("/")
async def home():
    logger.info("Home endpoint accessed.")
    return {"message": "Welcome to the Hand Gesture Classification Project"}


@app.get("/health")
async def health():
    if preprocess:
        if model:
            status="healthy"
        else:
            status="model file not loaded"      
    else:
        status="preprocessing file not loaded"    
    logger.info(f"Health check accessed. Status: {status}")
    return {"status": status}

prediction_latency = Summary('prediction_latency_seconds', 'Time spent on prediction')
landmark_mean = Gauge("input_landmark_mean", "Mean of landmark input values")
landmark_std = Gauge("input_landmark_std", "Std deviation of landmark input values")

@app.post("/predict")
@prediction_latency.time()
async def predict(request:LandmarkInput):
    if not preprocess:
        logger.error("Prediction request received but preprocessing file not loaded.")
        raise HTTPException(status_code=503, detail="preprocessing file not loaded")
    if not model:
        logger.error("Prediction request received but model not loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
      arr = np.array(request.landmarks)
      landmark_mean.set(np.mean(arr))
      landmark_std.set(np.std(arr))  
      logger.info(f"Prediction request received: features={request.landmarks}")
      pred=prediction(preprocess,model,[request.landmarks])
      logger.info(f"Prediction Result: {pred}")
      return {"prediction": pred}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail="Prediction failed")    



if __name__ == "__main__":   
    uvicorn.run(app, port=8000)
