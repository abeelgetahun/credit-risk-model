from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import pandas as pd

class CustomerFeatures(BaseModel):
    """Pydantic model for customer features input"""
    
    # RFM and transaction metrics
    Recency: float = Field(..., description="Customer recency score (normalized)")
    Frequency: float = Field(..., description="Customer frequency score (normalized)")
    TotalAmount: float = Field(..., description="Total transaction amount (normalized)")
    AvgAmount: float = Field(..., description="Average transaction amount (normalized)")
    StdAmount: float = Field(..., description="Standard deviation of amounts (normalized)")
    
    # Transaction counts
    TransactionCount_x: float = Field(..., description="Transaction count feature x (normalized)")
    TransactionCount_y: float = Field(..., description="Transaction count feature y (normalized)")
    
    # Amount statistics
    AmountSum: float = Field(..., description="Sum of amounts (normalized)")
    AmountMean: float = Field(..., description="Mean of amounts (normalized)")
    AmountStd: float = Field(..., description="Standard deviation of amounts (normalized)")
    AmountMin: float = Field(..., description="Minimum amount (normalized)")
    AmountMax: float = Field(..., description="Maximum amount (normalized)")
    AmountCount: float = Field(..., description="Count of amounts (normalized)")
    
    # Value statistics
    ValueSum: float = Field(..., description="Sum of values (normalized)")
    ValueMean: float = Field(..., description="Mean of values (normalized)")
    ValueStd: float = Field(..., description="Standard deviation of values (normalized)")
    ValueMin: float = Field(..., description="Minimum value (normalized)")
    ValueMax: float = Field(..., description="Maximum value (normalized)")
    
    # Channel and product features
    UniqueChannels: float = Field(..., description="Number of unique channels (normalized)")
    UniqueProductCategories: float = Field(..., description="Number of unique product categories (normalized)")
    UniqueProviders: float = Field(..., description="Number of unique providers (normalized)")
    UniqueProducts: float = Field(..., description="Number of unique products (normalized)")
    
    # Fraud and risk features
    TotalFraudCases: float = Field(..., description="Total fraud cases (normalized)")
    FraudRate: float = Field(..., description="Fraud rate (normalized)")
    
    # Transaction behavior
    AvgTransactionValue: float = Field(..., description="Average transaction value (normalized)")
    AmountValueRatio: float = Field(..., description="Amount to value ratio (normalized)")
    TransactionConsistency: float = Field(..., description="Transaction consistency score (normalized)")
    
    # Temporal features
    AvgTransactionHour: float = Field(..., description="Average transaction hour (normalized)")
    StdTransactionHour: float = Field(..., description="Standard deviation of transaction hours (normalized)")
    AvgTransactionDay: float = Field(..., description="Average transaction day (normalized)")
    StdTransactionDay: float = Field(..., description="Standard deviation of transaction days (normalized)")
    UniqueMonthsActive: float = Field(..., description="Number of unique months active (normalized)")
    AvgDayOfWeek: float = Field(..., description="Average day of week (normalized)")
    StdDayOfWeek: float = Field(..., description="Standard deviation of day of week (normalized)")
    WeekendTransactionRatio: float = Field(..., description="Weekend transaction ratio (normalized)")
    ActivitySpanDays: float = Field(..., description="Activity span in days (normalized)")
    TemporalTransactionCount: float = Field(..., description="Temporal transaction count (normalized)")
    
    class Config:
        schema_extra = {
            "example": {
                "Recency": 1.93760471589899,
                "Frequency": -0.25345907179094945,
                "TotalAmount": -0.06689055545071151,
                "AvgAmount": -0.15336428720432832,
                "StdAmount": -0.14043246692342928,
                "TransactionCount_x": -0.25345907179094945,
                "TransactionCount_y": -0.25345907179094945,
                "AmountSum": -0.06689055545071151,
                "AmountMean": -0.15336428770238522,
                "AmountStd": -0.14043246609822269,
                "AmountMin": -0.16153194237806467,
                "AmountMax": -0.16908050779935996,
                "AmountCount": -0.25345907179094945,
                "ValueSum": -0.08952358110188405,
                "ValueMean": -0.052297016997452796,
                "ValueStd": -0.13150788249858814,
                "ValueMin": 0.013620090533661311,
                "ValueMax": -0.12036213555198844,
                "UniqueChannels": -1.4047492209265093,
                "UniqueProductCategories": -1.1539767156489633,
                "UniqueProviders": -1.3827371298037798,
                "UniqueProducts": -1.115174679296779,
                "TotalFraudCases": -0.06661719721291895,
                "FraudRate": -0.08609596254315377,
                "AvgTransactionValue": -0.052297016997452796,
                "AmountValueRatio": -4.440728211016137,
                "TransactionConsistency": 1.8558371555723505,
                "AvgTransactionHour": 0.8832837898733554,
                "StdTransactionHour": -1.021754295182398,
                "AvgTransactionDay": 0.7205946864323469,
                "StdTransactionDay": 0.7705450391504929,
                "UniqueMonthsActive": 0.8942099132610634,
                "AvgDayOfWeek": -0.8236531921057049,
                "StdDayOfWeek": -0.8838273032894538,
                "WeekendTransactionRatio": -0.6369452487042843,
                "ActivitySpanDays": -0.7056868051389767,
                "TemporalTransactionCount": -0.25345907179094945
            }
        }

class LoanRequest(BaseModel):
    """Pydantic model for loan request"""
    customer_features: CustomerFeatures
    requested_amount: Optional[float] = Field(None, gt=0, description="Requested loan amount")
    
    @validator('requested_amount')
    def validate_amount(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Requested amount must be positive')
        return v

class RiskPredictionResponse(BaseModel):
    """Response model for risk prediction"""
    risk_probability: float = Field(..., ge=0, le=1, description="Probability of default (0-1)")
    risk_category: str = Field(..., description="Risk category (Low/Medium/High Risk)")
    
class CreditScoreResponse(BaseModel):
    """Response model for credit score"""
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    risk_probability: float = Field(..., ge=0, le=1, description="Risk probability")
    risk_category: str = Field(..., description="Risk category")

class LoanRecommendation(BaseModel):
    """Model for loan recommendation"""
    approval_status: str = Field(..., description="Loan approval status")
    recommended_amount: float = Field(..., gt=0, description="Recommended loan amount")
    max_amount: float = Field(..., gt=0, description="Maximum loan amount")
    recommended_duration_months: int = Field(..., gt=0, description="Recommended duration in months")
    interest_rate: float = Field(..., ge=0, description="Annual interest rate")
    monthly_payment: float = Field(..., gt=0, description="Monthly payment amount")

class ComprehensiveAssessmentResponse(BaseModel):
    """Complete credit risk assessment response"""
    risk_probability: float = Field(..., ge=0, le=1, description="Risk probability")
    risk_category: str = Field(..., description="Risk category")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score")
    loan_recommendations: LoanRecommendation = Field(..., description="Loan recommendations")

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    timestamp: str = Field(..., description="Current timestamp")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Error timestamp")