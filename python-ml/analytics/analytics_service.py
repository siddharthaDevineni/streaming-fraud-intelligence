"""
Analytics Service — PostgreSQL Query API
=========================================
Separate FastAPI microservice on port 8001 for SQL-based fraud analytics.

Endpoints:
- GET /fraud-patterns              → Fraud pattern aggregations
- GET /analyst-performance         → Analyst review metrics
- GET /model-metrics               → Model performance comparison
- GET /transaction-history/:id     → Customer transaction history
- GET /fraud-decisions/recent      → Recent fraud decisions
- GET /health                      → Database health check

Usage:
    python -m uvicorn analytics.analytics_service:app --host 0.0.0.0 --port 8001
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import structlog

from config import settings
from db.postgres_client import get_postgres_client

logger = structlog.get_logger()

# ═══════════════════════════════════════════════════════════════════════════
# FastAPI Application
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Fraud Detection Analytics API",
    description="PostgreSQL-based analytics for fraud detection audit trail",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration (for Streamlit dashboard, frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Postgres client
try:
    postgres_client = get_postgres_client(settings.postgres_url)
    logger.info("analytics_service_started", port=8001)
except Exception as e:
    logger.error("postgres_init_failed", error=str(e))
    postgres_client = None

# ═══════════════════════════════════════════════════════════════════════════
# Response Models
# ═══════════════════════════════════════════════════════════════════════════

class FraudPatternSummary(BaseModel):
    fraud_pattern: str
    total_cases: int
    avg_ml_score: float
    avg_llm_confidence: float
    blocked_count: int
    review_count: int
    rag_assisted_count: int
    first_seen: Optional[datetime]
    last_seen: Optional[datetime]


class AnalystPerformance(BaseModel):
    analyst_id: str
    total_reviews: int
    avg_review_time_sec: float
    confirmed_fraud: int
    false_positives: int
    escalated: int
    first_review: Optional[datetime]
    last_review: Optional[datetime]


class ModelMetrics(BaseModel):
    model_type: str
    total_predictions: int
    avg_score: float
    avg_latency_ms: float
    accuracy: Optional[float]
    first_prediction: Optional[datetime]
    last_prediction: Optional[datetime]


class TransactionHistory(BaseModel):
    transaction_id: str
    customer_id: str
    amount: float
    merchant_id: Optional[str]
    merchant_category: Optional[str]
    timestamp: datetime
    decision: Optional[str]
    fraud_pattern: Optional[str]
    llm_confidence: Optional[float]
    ml_score: Optional[float]


class FraudDecisionDetail(BaseModel):
    id: int
    transaction_id: str
    ml_score: Optional[float]
    llm_confidence: Optional[float]
    fraud_pattern: Optional[str]
    decision: Optional[str]
    reasoning: Optional[str]
    rag_context_used: bool
    decided_at: datetime


class HealthCheck(BaseModel):
    status: str
    database: str
    timestamp: datetime


# ═══════════════════════════════════════════════════════════════════════════
# Analytics Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Database connectivity health check."""
    if not postgres_client:
        raise HTTPException(status_code=503, detail="PostgreSQL client not initialized")

    db_healthy = postgres_client.health_check()

    return HealthCheck(
        status="healthy" if db_healthy else "degraded",
        database="connected" if db_healthy else "disconnected",
        timestamp=datetime.utcnow()
    )


@app.get("/fraud-patterns", response_model=List[Dict[str, Any]])
async def get_fraud_patterns():
    """
    Get aggregated fraud pattern statistics.

    Returns fraud cases grouped by pattern type with metrics:
    - Total cases per pattern
    - Average ML/LLM scores
    - Block vs review distribution
    - RAG-assisted detection count
    """
    if not postgres_client:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        patterns = postgres_client.get_fraud_pattern_summary()
        logger.info("fraud_patterns_retrieved", count=len(patterns))
        return patterns
    except Exception as e:
        logger.error("fraud_patterns_query_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/analyst-performance", response_model=List[Dict[str, Any]])
async def get_analyst_performance():
    """
    Get analyst performance metrics.

    Returns per-analyst statistics:
    - Total cases reviewed
    - Average review time (SLA tracking)
    - Decision distribution (confirmed fraud, false positives, escalated)
    """
    if not postgres_client:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        performance = postgres_client.get_analyst_performance()
        logger.info("analyst_performance_retrieved", count=len(performance))
        return performance
    except Exception as e:
        logger.error("analyst_performance_query_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/model-metrics", response_model=List[Dict[str, Any]])
async def get_model_metrics():
    """
    Compare model performance across XGBoost, River, LLM ensemble.

    Returns:
    - Prediction counts
    - Average scores
    - Latency (SLA tracking)
    - Accuracy (requires analyst feedback for ground truth)
    """
    if not postgres_client:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        metrics = postgres_client.get_model_performance_comparison()
        logger.info("model_metrics_retrieved", count=len(metrics))
        return metrics
    except Exception as e:
        logger.error("model_metrics_query_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/transaction-history/{customer_id}", response_model=List[Dict[str, Any]])
async def get_transaction_history(
    customer_id: str,
    limit: int = Query(default=50, ge=1, le=500, description="Number of transactions to return")
):
    """
    Get transaction history for a customer with fraud decisions.

    Joins transactions with fraud_decisions for complete audit trail.
    Useful for investigation, customer risk profiling, pattern analysis.
    """
    if not postgres_client:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        history = postgres_client.get_transaction_history(customer_id, limit)
        logger.info("transaction_history_retrieved",
                   customer_id=customer_id,
                   count=len(history))
        return history
    except Exception as e:
        logger.error("transaction_history_query_failed",
                    customer_id=customer_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/fraud-decisions/recent")
async def get_recent_fraud_decisions(
    limit: int = Query(default=100, ge=1, le=1000, description="Number of decisions to return"),
    pattern: Optional[str] = Query(default=None, description="Filter by fraud pattern")
):
    """
    Get recent fraud decisions with optional pattern filtering.

    Returns:
    - Transaction ID, ML score, LLM confidence
    - Fraud pattern classification
    - Decision (BLOCK/REVIEW/APPROVE)
    - RAG context usage
    """
    if not postgres_client:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        if pattern:
            decisions = postgres_client.get_fraud_decisions_by_pattern(pattern, limit)
        else:
            # Query all recent decisions (direct SQL)
            with postgres_client.session_scope() as session:
                from sqlalchemy import text
                result = session.execute(text("""
                    SELECT
                        id,
                        transaction_id,
                        ml_score,
                        llm_confidence,
                        fraud_pattern,
                        decision,
                        reasoning,
                        rag_context_used,
                        decided_at
                    FROM fraud_decisions
                    ORDER BY decided_at DESC
                    LIMIT :limit
                """), {"limit": limit})
                decisions = [dict(row._mapping) for row in result]

        logger.info("recent_decisions_retrieved", count=len(decisions), pattern=pattern)
        return decisions
    except Exception as e:
        logger.error("recent_decisions_query_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/stats/summary")
async def get_summary_stats():
    """
    Get overall system statistics.

    Returns high-level metrics:
    - Total transactions processed
    - Total fraud decisions
    - Total case reviews
    - Decision distribution
    """
    if not postgres_client:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        with postgres_client.session_scope() as session:
            from sqlalchemy import text

            # Get counts
            txn_count = session.execute(text("SELECT COUNT(*) FROM transactions")).scalar()
            decision_count = session.execute(text("SELECT COUNT(*) FROM fraud_decisions")).scalar()
            review_count = session.execute(text("SELECT COUNT(*) FROM case_reviews")).scalar()

            # Get decision distribution
            decision_dist = session.execute(text("""
                SELECT decision, COUNT(*) as count
                FROM fraud_decisions
                WHERE decision IS NOT NULL
                GROUP BY decision
            """))
            decision_distribution = {row[0]: row[1] for row in decision_dist}

            # Get pattern distribution (top 10)
            pattern_dist = session.execute(text("""
                SELECT fraud_pattern, COUNT(*) as count
                FROM fraud_decisions
                WHERE fraud_pattern IS NOT NULL
                GROUP BY fraud_pattern
                ORDER BY count DESC
                LIMIT 10
            """))
            top_patterns = [{"pattern": row[0], "count": row[1]} for row in pattern_dist]

        return {
            "total_transactions": txn_count,
            "total_fraud_decisions": decision_count,
            "total_case_reviews": review_count,
            "decision_distribution": decision_distribution,
            "top_fraud_patterns": top_patterns,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.error("summary_stats_query_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════
# Main (for local development)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
