from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class Transaction(BaseModel):
    transactionId: str
    customerId: str
    amount: float
    currency: str
    merchantId: str
    merchantCategory: str
    location: str
    timestamp: datetime
    metadata: dict


class CustomerProfile(BaseModel):
    customerId: str
    averageTransactionAmount: float
    dailySpendingLimit: float
    transactionCategories: list[str]
    primaryLocation: str
    riskLevel: str


class EnrichedTransaction(BaseModel):
    transaction: Transaction
    customerProfile: Optional[CustomerProfile] = None
    velocityCount: Optional[int] = None


class MLPrediction(BaseModel):
    transactionId: str
    customerId: str
    mlFraudScore: float
    lstmSequenceScore: float
    combinedScore: float
    modelVersion: str
    shapExplanation: dict
    featuresUsed: dict
    ragContext: dict = {}
    inferenceLatencyMs: int
    timestamp: datetime


class AnalystFeedback(BaseModel):
    transactionId: str
    predictedFraud: bool
    actualFraud: Optional[bool] = None
    confidence: float
    agentConsensus: int
    timestamp: datetime