from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fix: Import the preprocessor instance, not data_preprocessing
from src.preprocessing import preprocessor
from src.model_training import SentimentModel

# Initialize FastAPI app
app = FastAPI(
    title="Women's Clothing Reviews Sentiment Analysis API",
    description="API for predicting sentiment from clothing reviews",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = SentimentModel()
try:
    model.load_model(
        '../models/logistic_regression.pkl',
        '../models/tfidf_vectorizer.pkl'
    )
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Pydantic models
class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Review text to analyze")
    
class ReviewBatchRequest(BaseModel):
    texts: List[str] = Field(..., description="List of review texts to analyze")

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: dict
    text: str

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    total_processed: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    total_predictions: Optional[int] = 0

# Statistics
predictions_count = 0

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None and hasattr(model, 'is_trained') and model.is_trained,
        total_predictions=predictions_count
    )

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(review: ReviewRequest):
    """Predict sentiment for a single review"""
    global predictions_count
    
    if model is None or not hasattr(model, 'is_trained') or not model.is_trained:
        raise HTTPException(status_code=503, detail="Model not loaded or not trained")
    
    try:
        # Clean text using preprocessor
        cleaned_text = preprocessor.clean_text(review.text)
        
        # Predict
        result = model.predict(cleaned_text)
        predictions_count += 1
        
        return SentimentResponse(
            sentiment=result['sentiment'],
            confidence=result['confidence'],
            text=review.text
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchSentimentResponse)
async def predict_batch(reviews: ReviewBatchRequest):
    """Predict sentiment for multiple reviews"""
    global predictions_count
    
    if model is None or not hasattr(model, 'is_trained') or not model.is_trained:
        raise HTTPException(status_code=503, detail="Model not loaded or not trained")
    
    try:
        # Clean texts
        cleaned_texts = [preprocessor.clean_text(text) for text in reviews.texts]
        
        # Predict
        results = model.predict_batch(cleaned_texts)
        predictions_count += len(results)
        
        # Format response
        formatted_results = []
        for i, result in enumerate(results):
            formatted_results.append(
                SentimentResponse(
                    sentiment=result['sentiment'],
                    confidence=result['confidence'],
                    text=reviews.texts[i]
                )
            )
        
        return BatchSentimentResponse(
            results=formatted_results,
            total_processed=len(formatted_results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None or not hasattr(model, 'is_trained') or not model.is_trained:
        raise HTTPException(status_code=503, detail="Model not loaded or not trained")
    
    return {
        "classes": model.model.classes_.tolist(),
        "vectorizer_params": {
            "max_features": model.vectorizer.max_features,
            "ngram_range": model.vectorizer.ngram_range
        },
        "model_type": type(model.model).__name__
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)