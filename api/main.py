from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import sys
import os

# Ensure the root directory is in the path to import src.predictor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predictor import PredictorService

predictor_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor_service
    print("Initializing PredictorService...")
    predictor_service = PredictorService(models_path="models")
    yield
    predictor_service = None

app = FastAPI(title="News Topic Classifier API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    text: str

@app.get("/health")
async def health_check():
    return {"status": "API is running"}

@app.post("/predict")
async def predict_topic(request: PredictionRequest):
    global predictor_service
    if predictor_service is None:
        return {"error": "PredictorService is not initialized"}
    
    result = predictor_service.predict(request.text)
    return result
