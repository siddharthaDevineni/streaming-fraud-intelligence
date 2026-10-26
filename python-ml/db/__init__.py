"""
Database layer for PostgreSQL persistence.

Provides SQLAlchemy models and database client for fraud detection audit trail.
"""

from .models import (
    Transaction,
    Customer,
    FraudDecision,
    CaseReview,
    ModelPrediction,
    FeatureStats,
    Base
)
from .postgres_client import PostgresClient, get_postgres_client

__all__ = [
    'Transaction',
    'Customer',
    'FraudDecision',
    'CaseReview',
    'ModelPrediction',
    'FeatureStats',
    'Base',
    'PostgresClient',
    'get_postgres_client'
]
