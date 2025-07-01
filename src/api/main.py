from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import sys
import os
from typing import List, Dict, Any

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict import CreditRiskPredictor
from api.pydantic_models import (
    CustomerFeatures, LoanRequest, RiskPredictionResponse,
    CreditScoreResponse, ComprehensiveAssessmentResponse,
    HealthCheckResponse, ErrorResponse
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Assessment API",
    description="API for credit risk assessment and loan recommendations for Bati Bank",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model when the API starts"""
    global predictor
    try:
        logger.info("Loading credit risk model...")
        predictor = CreditRiskPredictor()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        # Don't raise exception here to allow API to start even if model fails to load

def get_predictor():
    """Dependency to get the predictor instance"""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )
    return predictor

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    return HealthCheckResponse(
        status="healthy",
        model_loaded=predictor is not None,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict/risk", response_model=RiskPredictionResponse)
async def predict_risk(
    customer_features: CustomerFeatures,
    predictor_instance: CreditRiskPredictor = Depends(get_predictor)
):
    """
    Predict credit risk probability for a customer
    
    Args:
        customer_features: Customer feature data
        
    Returns:
        Risk probability and category
    """
    try:
        logger.info("Processing risk prediction request")
        
        # Convert to dictionary for prediction
        features_dict = customer_features.dict()
        
        # Get prediction
        risk_prob, risk_category = predictor_instance.predict_risk_probability(features_dict)
        
        # Handle array results
        if isinstance(risk_prob, np.ndarray):
            risk_prob = float(risk_prob[0])
        if isinstance(risk_category, list):
            risk_category = risk_category[0]
        
        return RiskPredictionResponse(
            risk_probability=risk_prob,
            risk_category=risk_category
        )
        
    except Exception as e:
        logger.error(f"Error in risk prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing prediction: {str(e)}"
        )

@app.post("/predict/credit-score", response_model=CreditScoreResponse)
async def predict_credit_score(
    customer_features: CustomerFeatures,
    predictor_instance: CreditRiskPredictor = Depends(get_predictor)
):
    """
    Calculate credit score for a customer
    
    Args:
        customer_features: Customer feature data
        
    Returns:
        Credit score, risk probability, and risk category
    """
    try:
        logger.info("Processing credit score request")
        
        # Convert to dictionary for prediction
        features_dict = customer_features.dict()
        
        # Get risk prediction first
        risk_prob, risk_category = predictor_instance.predict_risk_probability(features_dict)
        
        # Calculate credit score
        credit_score = predictor_instance.predict_credit_score(risk_prob)
        
        # Handle array results
        if isinstance(risk_prob, np.ndarray):
            risk_prob = float(risk_prob[0])
        if isinstance(risk_category, list):
            risk_category = risk_category[0]
        if isinstance(credit_score, list):
            credit_score = int(credit_score[0])
        
        return CreditScoreResponse(
            credit_score=credit_score,
            risk_probability=risk_prob,
            risk_category=risk_category
        )
        
    except Exception as e:
        logger.error(f"Error in credit score calculation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating credit score: {str(e)}"
        )

@app.post("/assess", response_model=ComprehensiveAssessmentResponse)
async def comprehensive_assessment(
    loan_request: LoanRequest,
    predictor_instance: CreditRiskPredictor = Depends(get_predictor)
):
    """
    Perform comprehensive credit risk assessment with loan recommendations
    
    Args:
        loan_request: Loan request with customer features and optional amount
        
    Returns:
        Complete assessment including risk, credit score, and loan recommendations
    """
    try:
        logger.info("Processing comprehensive assessment request")
        
        # Convert to dictionary for prediction
        features_dict = loan_request.customer_features.dict()
        
        # Get comprehensive assessment
        assessment = predictor_instance.comprehensive_assessment(
            features_dict, 
            requested_amount=loan_request.requested_amount
        )
        
        return ComprehensiveAssessmentResponse(**assessment)
        
    except Exception as e:
        logger.error(f"Error in comprehensive assessment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing assessment: {str(e)}"
        )

@app.post("/batch/assess")
async def batch_assessment(
    customers: List[CustomerFeatures],
    predictor_instance: CreditRiskPredictor = Depends(get_predictor)
):
    """
    Perform batch assessment for multiple customers
    
    Args:
        customers: List of customer feature data
        
    Returns:
        List of assessments
    """
    try:
        logger.info(f"Processing batch assessment for {len(customers)} customers")
        
        results = []
        for i, customer in enumerate(customers):
            try:
                features_dict = customer.dict()
                assessment = predictor_instance.comprehensive_assessment(features_dict)
                results.append({
                    "customer_index": i,
                    "assessment": assessment,
                    "status": "success"
                })
            except Exception as e:
                logger.error(f"Error processing customer {i}: {str(e)}")
                results.append({
                    "customer_index": i,
                    "error": str(e),
                    "status": "error"
                })
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Error in batch assessment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing batch assessment: {str(e)}"
        )

@app.get("/model/info")
async def model_info(predictor_instance: CreditRiskPredictor = Depends(get_predictor)):
    """Get information about the loaded model"""
    try:
        return {
            "model_name": predictor_instance.model_name,
            "model_version": predictor_instance.model_version,
            "uses_scaled_features": predictor_instance.use_scaled_features,
            "model_type": type(predictor_instance.model).__name__ if predictor_instance.model else None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving model information: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Credit Risk Assessment API",
        "version": "1.0.0",
        "description": "API for credit risk assessment and loan recommendations",
        "endpoints": {
            "health": "/health",
            "predict_risk": "/predict/risk",
            "predict_credit_score": "/predict/credit-score",
            "comprehensive_assessment": "/assess",
            "batch_assessment": "/batch/assess",
            "model_info": "/model/info",
            "documentation": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)