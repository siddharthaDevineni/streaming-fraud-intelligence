"""
SQLAlchemy ORM models for fraud detection PostgreSQL schema.

Maps Python classes to database tables for type-safe persistence.
Matches schema defined in postgres/init.sql.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    Column, String, Numeric, Integer, DateTime, Boolean, Text,
    ForeignKey, CheckConstraint, Index, text
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Transaction(Base):
    """
    Core transaction audit trail.

    Immutable record of every transaction processed through the fraud detection system.
    Includes enrichment data from Kafka Streams (velocity, customer profile).
    """
    __tablename__ = 'transactions'

    transaction_id = Column(String(50), primary_key=True)
    customer_id = Column(String(50), nullable=False, index=True)
    amount = Column(Numeric(12, 2), nullable=False)
    merchant_id = Column(String(50), index=True)
    merchant_category = Column(String(50))
    location = Column(String(100))
    device_id = Column(String(50))
    timestamp = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Enrichment fields from Kafka Streams
    velocity_5min = Column(Integer, default=0)
    avg_transaction_amount = Column(Numeric(12, 2))
    account_age_days = Column(Integer)

    # Relationships
    fraud_decisions = relationship('FraudDecision', back_populates='transaction')
    case_reviews = relationship('CaseReview', back_populates='transaction')

    __table_args__ = (
        CheckConstraint('amount >= 0', name='chk_amount_positive'),
        CheckConstraint('timestamp <= NOW()', name='chk_timestamp_not_future'),
        Index('idx_transactions_customer_time', 'customer_id', text('timestamp DESC')),
    )

    def __repr__(self):
        return f"<Transaction(id={self.transaction_id}, customer={self.customer_id}, amount={self.amount})>"


class Customer(Base):
    """
    Customer profile with persistent storage and versioning.

    Dual-write pattern: synced with Kafka KTable for fault tolerance.
    Version column enables optimistic locking for concurrent updates.
    """
    __tablename__ = 'customers'

    customer_id = Column(String(50), primary_key=True)
    avg_transaction_amount = Column(Numeric(12, 2))
    typical_merchant_category = Column(String(50))
    account_age_days = Column(Integer)
    risk_score = Column(Numeric(5, 2))
    total_transactions = Column(Integer, default=0)
    total_fraud_cases = Column(Integer, default=0)
    last_transaction_at = Column(DateTime)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    version = Column(Integer, default=1)

    __table_args__ = (
        CheckConstraint('account_age_days >= 0', name='chk_account_age_positive'),
        CheckConstraint('risk_score BETWEEN 0 AND 100', name='chk_risk_score_range'),
        CheckConstraint('total_fraud_cases <= total_transactions', name='chk_fraud_ratio'),
        Index('idx_customers_risk_score', text('risk_score DESC')),
    )

    def __repr__(self):
        return f"<Customer(id={self.customer_id}, risk_score={self.risk_score})>"


class FraudDecision(Base):
    """
    Complete fraud decision audit trail with ML + LLM context.

    Stores:
    - XGBoost ML score + SHAP feature importances
    - LLM ensemble confidence + reasoning + fraud pattern
    - RAG retrieval metadata (similarity, cases used)
    - Final routing decision (BLOCK/REVIEW/APPROVE)

    Immutable audit trail (compliance requirement, 7+ year retention).
    """
    __tablename__ = 'fraud_decisions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(50), ForeignKey('transactions.transaction_id'), nullable=False)

    # ML layer outputs
    ml_score = Column(Numeric(5, 4))
    ml_model_type = Column(String(20), default='xgboost')
    shap_top_features = Column(JSONB)  # [{feature: str, value: float}, ...]

    # LLM ensemble outputs
    llm_confidence = Column(Numeric(5, 2))
    fraud_pattern = Column(String(50), index=True)
    reasoning = Column(Text)
    recommendation = Column(Text)

    # RAG context metadata
    rag_context_used = Column(Boolean, default=False)
    rag_similarity_score = Column(Numeric(5, 4))
    rag_cases_retrieved = Column(Integer, default=0)

    # Final decision
    decision = Column(String(20), index=True)
    decided_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    transaction = relationship('Transaction', back_populates='fraud_decisions')
    case_reviews = relationship('CaseReview', back_populates='fraud_decision')

    __table_args__ = (
        CheckConstraint('ml_score BETWEEN 0 AND 1', name='chk_ml_score_range'),
        CheckConstraint('llm_confidence BETWEEN 0 AND 100', name='chk_llm_confidence_range'),
        CheckConstraint("decision IN ('BLOCK', 'REVIEW', 'APPROVE')", name='chk_decision_values'),
        CheckConstraint(
            "(rag_context_used = TRUE AND rag_similarity_score IS NOT NULL) OR (rag_context_used = FALSE)",
            name='chk_rag_consistency'
        ),
        Index('idx_fraud_decisions_shap', 'shap_top_features', postgresql_using='gin'),
    )

    def __repr__(self):
        return f"<FraudDecision(txn={self.transaction_id}, decision={self.decision}, confidence={self.llm_confidence})>"


class CaseReview(Base):
    """
    Analyst workflow tracking and investigation notes.

    Stores analyst feedback which feeds:
    - ChromaDB embeddings (RAG retrieval for future cases)
    - River online learning (continuous model updates)

    Tracks SLA metrics (time_to_review_seconds).
    """
    __tablename__ = 'case_reviews'

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(50), ForeignKey('transactions.transaction_id'), nullable=False)
    analyst_id = Column(String(50), index=True)
    review_outcome = Column(String(30), index=True)
    notes = Column(Text)
    time_to_review_seconds = Column(Integer)
    reviewed_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Link to original fraud decision
    fraud_decision_id = Column(Integer, ForeignKey('fraud_decisions.id'))

    # Investigation metadata
    evidence_sources = Column(JSONB)
    confidence_level = Column(String(20))

    # Relationships
    transaction = relationship('Transaction', back_populates='case_reviews')
    fraud_decision = relationship('FraudDecision', back_populates='case_reviews')

    __table_args__ = (
        CheckConstraint(
            "review_outcome IN ('CONFIRMED_FRAUD', 'FALSE_POSITIVE', 'ESCALATED', 'LEGITIMATE')",
            name='chk_review_outcome_values'
        ),
        CheckConstraint('time_to_review_seconds >= 0', name='chk_review_time_positive'),
        CheckConstraint(
            "confidence_level IN ('LOW', 'MEDIUM', 'HIGH')",
            name='chk_confidence_level_values'
        ),
    )

    def __repr__(self):
        return f"<CaseReview(txn={self.transaction_id}, outcome={self.review_outcome}, analyst={self.analyst_id})>"


class ModelPrediction(Base):
    """
    Model performance tracking and drift detection.

    Per-model prediction storage for:
    - XGBoost (fast, feature-engineered)
    - River SRP (online learning)
    - LLM Ensemble (multi-agent reasoning)

    Enables precision/recall analysis and model comparison.
    """
    __tablename__ = 'model_predictions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(50), nullable=False, index=True)
    model_type = Column(String(30), index=True)
    prediction_score = Column(Numeric(5, 4))
    top_features = Column(JSONB)
    prediction_time_ms = Column(Integer)
    predicted_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Version tracking for A/B testing
    model_version = Column(String(20))

    # Ground truth comparison (populated after analyst feedback)
    actual_label = Column(Boolean)
    prediction_correct = Column(Boolean)
    updated_at = Column(DateTime)

    __table_args__ = (
        CheckConstraint(
            "model_type IN ('xgboost', 'river_srp', 'llm_ensemble')",
            name='chk_model_type_values'
        ),
        CheckConstraint('prediction_score BETWEEN 0 AND 1', name='chk_prediction_score_range'),
        CheckConstraint('prediction_time_ms >= 0', name='chk_prediction_time_positive'),
    )

    def __repr__(self):
        return f"<ModelPrediction(txn={self.transaction_id}, model={self.model_type}, score={self.prediction_score})>"


class FeatureStats(Base):
    """
    Feature drift monitoring and data quality metrics.

    Tracks statistical distribution of features over time windows.
    Compare production stats vs training distribution to detect drift.
    """
    __tablename__ = 'feature_stats'

    id = Column(Integer, primary_key=True, autoincrement=True)
    feature_name = Column(String(100), nullable=False, index=True)
    stat_type = Column(String(20), index=True)
    stat_value = Column(Numeric(12, 4))
    sample_count = Column(Integer)
    window_start = Column(DateTime, nullable=False)
    window_end = Column(DateTime, nullable=False)
    computed_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        CheckConstraint(
            "stat_type IN ('mean', 'stddev', 'median', 'p25', 'p50', 'p75', 'p95', 'p99', 'min', 'max', 'null_rate')",
            name='chk_stat_type_values'
        ),
        CheckConstraint('window_end > window_start', name='chk_window_order'),
        Index('uq_feature_stat_window', 'feature_name', 'stat_type', 'window_start', unique=True),
        Index('idx_feature_stats_name_time', 'feature_name', text('window_start DESC')),
    )

    def __repr__(self):
        return f"<FeatureStats(feature={self.feature_name}, stat={self.stat_type}, value={self.stat_value})>"
