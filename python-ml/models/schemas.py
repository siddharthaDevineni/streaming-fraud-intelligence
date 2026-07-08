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

class AgentInsightOutput(BaseModel):
    """Matches AgentInsight.java record exactly — Java deserializes this."""
    agentType: str
    agentName: str
    analysis: str
    riskScore: float
    confidence: float
    reasoning: str
    recommendation: str
    timestamp: str  # ISO format string — Java LocalDateTime parses this

class FraudDecisionOutput(BaseModel):
    """
    Matches FraudDecision.java record exactly.
    Python publishes this to fraud-decisions topic.
    Java reads it via KTable join and routes to output topics.
    """
    transactionId: str
    isFraudulent: bool
    confidenceScore: float
    primaryReason: str
    detailedExplanation: str
    agentInsights: list[AgentInsightOutput]
    riskFactors: dict = {}
    analyzedAt: str  # ISO format string — Java LocalDateTime parses this
    fraudPattern: str = "unknown"