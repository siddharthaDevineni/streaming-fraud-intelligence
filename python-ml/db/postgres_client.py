"""
PostgreSQL client with connection pooling and repository pattern.

Provides clean interface for fraud detection persistence operations.
Thread-safe session management with context managers.
"""

import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, List, Dict, Any
from decimal import Decimal

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from .models import (
    Transaction, Customer, FraudDecision, CaseReview,
    ModelPrediction, FeatureStats, Base
)

logger = logging.getLogger(__name__)


class PostgresClient:
    """
    PostgreSQL database client with connection pooling.

    Usage:
        client = PostgresClient("postgresql://user:pass@host:5432/db")

        # Automatic session management
        client.save_transaction(txn_data)

        # Manual session control
        with client.session_scope() as session:
            session.add(my_object)
            session.commit()
    """

    def __init__(
        self,
        connection_string: str,
        pool_size: int = 10,
        max_overflow: int = 20,
        echo: bool = False
    ):
        """
        Initialize database connection with pooling.

        Args:
            connection_string: PostgreSQL connection URL
            pool_size: Number of connections to maintain in pool
            max_overflow: Max additional connections beyond pool_size
            echo: Log SQL queries (useful for debugging)
        """
        self.engine = create_engine(
            connection_string,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,  # Verify connections before using
            echo=echo
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        logger.info(f"PostgreSQL client initialized with pool_size={pool_size}")

    @contextmanager
    def session_scope(self):
        """
        Provide transactional scope for database operations.

        Automatically commits on success, rolls back on exception.

        Usage:
            with client.session_scope() as session:
                session.add(my_object)
                # Auto-commits here if no exception
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error, rolling back: {e}")
            raise
        finally:
            session.close()

    def create_tables(self):
        """Create all tables (only for local dev/testing, not production)."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("All tables created successfully")

    # ═══════════════════════════════════════════════════════════════════════
    # Transaction Operations
    # ═══════════════════════════════════════════════════════════════════════

    def save_transaction(self, data: Dict[str, Any]) -> Transaction:
        """
        Save transaction to audit trail.

        Args:
            data: Dict with keys: transaction_id, customer_id, amount, merchant_id,
                  merchant_category, location, device_id, timestamp, velocity_5min,
                  avg_transaction_amount, account_age_days

        Returns:
            Transaction ORM object
        """
        with self.session_scope() as session:
            txn = Transaction(
                transaction_id=data['transaction_id'],
                customer_id=data['customer_id'],
                amount=Decimal(str(data['amount'])),
                merchant_id=data.get('merchant_id'),
                merchant_category=data.get('merchant_category'),
                location=data.get('location'),
                device_id=data.get('device_id'),
                timestamp=data['timestamp'],
                velocity_5min=data.get('velocity_5min', 0),
                avg_transaction_amount=Decimal(str(data['avg_transaction_amount'])) if data.get('avg_transaction_amount') else None,
                account_age_days=data.get('account_age_days')
            )
            session.add(txn)
            session.commit()
            session.refresh(txn)
            logger.debug(f"Saved transaction {txn.transaction_id}")
            return txn

    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """Retrieve transaction by ID."""
        with self.session_scope() as session:
            return session.query(Transaction).filter_by(transaction_id=transaction_id).first()

    def get_customer_transactions(
        self,
        customer_id: str,
        limit: int = 100
    ) -> List[Transaction]:
        """Get recent transactions for a customer."""
        with self.session_scope() as session:
            return session.query(Transaction)\
                .filter_by(customer_id=customer_id)\
                .order_by(Transaction.timestamp.desc())\
                .limit(limit)\
                .all()

    # ═══════════════════════════════════════════════════════════════════════
    # Customer Operations
    # ═══════════════════════════════════════════════════════════════════════

    def save_customer(self, data: Dict[str, Any]) -> Customer:
        """
        Save or update customer profile.

        Uses optimistic locking (version column) for concurrent updates.

        Args:
            data: Dict with keys: customer_id, avg_transaction_amount,
                  typical_merchant_category, account_age_days, risk_score,
                  total_transactions, total_fraud_cases, last_transaction_at
        """
        with self.session_scope() as session:
            existing = session.query(Customer).filter_by(
                customer_id=data['customer_id']
            ).first()

            if existing:
                # Update existing customer
                existing.avg_transaction_amount = Decimal(str(data.get('avg_transaction_amount', 0)))
                existing.typical_merchant_category = data.get('typical_merchant_category')
                existing.account_age_days = data.get('account_age_days')
                existing.risk_score = Decimal(str(data.get('risk_score', 0)))
                existing.total_transactions = data.get('total_transactions', existing.total_transactions)
                existing.total_fraud_cases = data.get('total_fraud_cases', existing.total_fraud_cases)
                existing.last_transaction_at = data.get('last_transaction_at')
                existing.last_updated = datetime.utcnow()
                existing.version += 1
                session.commit()
                session.refresh(existing)
                logger.debug(f"Updated customer {existing.customer_id} (version {existing.version})")
                return existing
            else:
                # Create new customer
                customer = Customer(
                    customer_id=data['customer_id'],
                    avg_transaction_amount=Decimal(str(data.get('avg_transaction_amount', 0))),
                    typical_merchant_category=data.get('typical_merchant_category'),
                    account_age_days=data.get('account_age_days'),
                    risk_score=Decimal(str(data.get('risk_score', 0))),
                    total_transactions=data.get('total_transactions', 0),
                    total_fraud_cases=data.get('total_fraud_cases', 0),
                    last_transaction_at=data.get('last_transaction_at')
                )
                session.add(customer)
                session.commit()
                session.refresh(customer)
                logger.debug(f"Created customer {customer.customer_id}")
                return customer

    def get_customer(self, customer_id: str) -> Optional[Customer]:
        """Retrieve customer profile by ID."""
        with self.session_scope() as session:
            return session.query(Customer).filter_by(customer_id=customer_id).first()

    # ═══════════════════════════════════════════════════════════════════════
    # Fraud Decision Operations
    # ═══════════════════════════════════════════════════════════════════════

    def save_fraud_decision(self, data: Dict[str, Any]) -> FraudDecision:
        """
        Save fraud decision to audit trail.

        Args:
            data: Dict with keys: transaction_id, ml_score, ml_model_type,
                  shap_top_features (list of dicts), llm_confidence, fraud_pattern,
                  reasoning, recommendation, rag_context_used, rag_similarity_score,
                  rag_cases_retrieved, decision
        """
        with self.session_scope() as session:
            decision = FraudDecision(
                transaction_id=data['transaction_id'],
                ml_score=Decimal(str(data.get('ml_score', 0))),
                ml_model_type=data.get('ml_model_type', 'xgboost'),
                shap_top_features=data.get('shap_top_features'),
                llm_confidence=Decimal(str(data.get('llm_confidence', 0))),
                fraud_pattern=data.get('fraud_pattern'),
                reasoning=data.get('reasoning'),
                recommendation=data.get('recommendation'),
                rag_context_used=data.get('rag_context_used', False),
                rag_similarity_score=Decimal(str(data.get('rag_similarity_score', 0))) if data.get('rag_similarity_score') else None,
                rag_cases_retrieved=data.get('rag_cases_retrieved', 0),
                decision=data.get('decision')
            )
            session.add(decision)
            session.commit()
            session.refresh(decision)
            logger.debug(f"Saved fraud decision for txn {decision.transaction_id}: {decision.decision}")
            return decision

    def get_fraud_decisions_by_pattern(
        self,
        pattern: str,
        limit: int = 100
    ) -> List[FraudDecision]:
        """Get recent fraud decisions for a specific pattern."""
        with self.session_scope() as session:
            return session.query(FraudDecision)\
                .filter_by(fraud_pattern=pattern)\
                .order_by(FraudDecision.decided_at.desc())\
                .limit(limit)\
                .all()

    def get_fraud_pattern_summary(self) -> List[Dict[str, Any]]:
        """
        Get aggregated fraud pattern statistics.

        Returns list of dicts with pattern, count, avg_ml_score, avg_llm_confidence.
        """
        with self.session_scope() as session:
            result = session.execute(text("""
                SELECT * FROM fraud_summary_by_pattern
            """))
            return [dict(row._mapping) for row in result]

    # ═══════════════════════════════════════════════════════════════════════
    # Case Review Operations
    # ═══════════════════════════════════════════════════════════════════════

    def save_case_review(self, data: Dict[str, Any]) -> CaseReview:
        """
        Save analyst case review.

        Args:
            data: Dict with keys: transaction_id, analyst_id, review_outcome,
                  notes, time_to_review_seconds, fraud_decision_id,
                  evidence_sources (dict), confidence_level
        """
        with self.session_scope() as session:
            review = CaseReview(
                transaction_id=data['transaction_id'],
                analyst_id=data.get('analyst_id'),
                review_outcome=data.get('review_outcome'),
                notes=data.get('notes'),
                time_to_review_seconds=data.get('time_to_review_seconds'),
                fraud_decision_id=data.get('fraud_decision_id'),
                evidence_sources=data.get('evidence_sources'),
                confidence_level=data.get('confidence_level')
            )
            session.add(review)
            session.commit()
            session.refresh(review)
            logger.debug(f"Saved case review for txn {review.transaction_id}: {review.review_outcome}")
            return review

    def get_analyst_performance(self) -> List[Dict[str, Any]]:
        """
        Get analyst performance metrics.

        Returns list of dicts with analyst_id, total_reviews, avg_review_time_sec,
        confirmed_fraud, false_positives, escalated.
        """
        with self.session_scope() as session:
            result = session.execute(text("""
                SELECT * FROM analyst_performance
            """))
            return [dict(row._mapping) for row in result]

    # ═══════════════════════════════════════════════════════════════════════
    # Model Prediction Operations
    # ═══════════════════════════════════════════════════════════════════════

    def save_model_prediction(self, data: Dict[str, Any]) -> ModelPrediction:
        """
        Save model prediction for performance tracking.

        Args:
            data: Dict with keys: transaction_id, model_type, prediction_score,
                  top_features (dict), prediction_time_ms, model_version
        """
        with self.session_scope() as session:
            prediction = ModelPrediction(
                transaction_id=data['transaction_id'],
                model_type=data['model_type'],
                prediction_score=Decimal(str(data.get('prediction_score', 0))),
                top_features=data.get('top_features'),
                prediction_time_ms=data.get('prediction_time_ms'),
                model_version=data.get('model_version')
            )
            session.add(prediction)
            session.commit()
            session.refresh(prediction)
            logger.debug(f"Saved {prediction.model_type} prediction for txn {prediction.transaction_id}")
            return prediction

    def update_prediction_ground_truth(
        self,
        transaction_id: str,
        model_type: str,
        actual_label: bool
    ):
        """
        Update prediction with ground truth after analyst feedback.

        Calculates prediction_correct field for accuracy tracking.
        """
        with self.session_scope() as session:
            prediction = session.query(ModelPrediction).filter_by(
                transaction_id=transaction_id,
                model_type=model_type
            ).first()

            if prediction:
                prediction.actual_label = actual_label
                prediction.prediction_correct = (
                    (prediction.prediction_score >= 0.5) == actual_label
                )
                prediction.updated_at = datetime.utcnow()
                session.commit()
                logger.debug(
                    f"Updated ground truth for {transaction_id}/{model_type}: "
                    f"correct={prediction.prediction_correct}"
                )

    def get_model_performance_comparison(self) -> List[Dict[str, Any]]:
        """
        Compare model performance metrics.

        Returns list of dicts with model_type, total_predictions, avg_score,
        avg_latency_ms, accuracy.
        """
        with self.session_scope() as session:
            result = session.execute(text("""
                SELECT * FROM model_performance_comparison
            """))
            return [dict(row._mapping) for row in result]

    # ═══════════════════════════════════════════════════════════════════════
    # Analytics Queries
    # ═══════════════════════════════════════════════════════════════════════

    def get_transaction_history(
        self,
        customer_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get transaction history with fraud decisions.

        Joins transactions with fraud_decisions for complete view.
        """
        with self.session_scope() as session:
            result = session.execute(text("""
                SELECT
                    t.transaction_id,
                    t.customer_id,
                    t.amount,
                    t.merchant_id,
                    t.merchant_category,
                    t.timestamp,
                    fd.decision,
                    fd.fraud_pattern,
                    fd.llm_confidence,
                    fd.ml_score
                FROM transactions t
                LEFT JOIN fraud_decisions fd ON t.transaction_id = fd.transaction_id
                WHERE t.customer_id = :customer_id
                ORDER BY t.timestamp DESC
                LIMIT :limit
            """), {"customer_id": customer_id, "limit": limit})
            return [dict(row._mapping) for row in result]

    def health_check(self) -> bool:
        """Test database connection."""
        try:
            with self.session_scope() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════
# Singleton Pattern (for service initialization)
# ═══════════════════════════════════════════════════════════════════════════

_postgres_client: Optional[PostgresClient] = None


def get_postgres_client(connection_string: str) -> PostgresClient:
    """
    Get or create PostgresClient singleton.

    Thread-safe initialization (uses module-level singleton).

    Usage:
        from db.postgres_client import get_postgres_client
        client = get_postgres_client(os.getenv('POSTGRES_URL'))
    """
    global _postgres_client
    if _postgres_client is None:
        _postgres_client = PostgresClient(connection_string)
        logger.info("PostgresClient singleton initialized")
    return _postgres_client
