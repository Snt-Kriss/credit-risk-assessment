from pydantic import BaseModel, Field
from typing import Optional

class CustomerFeatures(BaseModel):
    age: int= Field(..., ge=18, description="Customer age in years")
    job: int= Field(..., ge=0, description="Job category")
    credit_amount: int= Field(..., ge=0, description="Amount requested")
    duration: int= Field(..., ge=1, description="Duration of loan in months")

class PredictionResponse(BaseModel):
    risk: str= Field(..., description="Predicted risk category (Good/Bad)")
    probability: float= Field(..., description="Probability of being 'Good' class")