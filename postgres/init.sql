-- ═══════════════════════════════════════════════════════════════════════════
-- Streaming Fraud Intelligence - PostgreSQL Schema
-- ═══════════════════════════════════════════════════════════════════════════
--
-- Purpose: Persistent storage layer for:
--   1. Fraud decisions audit trail (compliance/regulatory)
--   2. Customer profile management (persistent CRUD)
--   3. Investigation case management (analyst workflow)
--   4. Model performance metrics (drift detection, accuracy tracking)
--   5. Long-term analytics (beyond Kafka retention)
--
-- Database: fraud_detection
-- User: fraud_user
-- Password: fraud_pass (docker-compose only, not for production)
--
-- ═══════════════════════════════════════════════════════════════════════════

-- Enable UUID extension for generating unique IDs
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ═══════════════════════════════════════════════════════════════════════════
-- Table 1: transactions
-- Core transaction audit trail with full context
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE transactions (
    transaction_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    amount DECIMAL(12, 2) NOT NULL CHECK (amount >= 0),
    merchant_id VARCHAR(50),
    merchant_category VARCHAR(50),
    location VARCHAR(100),
    device_id VARCHAR(50),
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),

    -- Additional enrichment fields from Kafka Streams
    velocity_5min INTEGER DEFAULT 0,
    avg_transaction_amount DECIMAL(12, 2),
    account_age_days INTEGER,

    CONSTRAINT chk_timestamp CHECK (timestamp <= NOW())
);

-- Indexes for common query patterns
CREATE INDEX idx_transactions_customer_time ON transactions(customer_id, timestamp DESC);
CREATE INDEX idx_transactions_merchant ON transactions(merchant_id);
CREATE INDEX idx_transactions_time ON transactions(timestamp DESC);
CREATE INDEX idx_transactions_amount ON transactions(amount);

COMMENT ON TABLE transactions IS 'Immutable audit trail of all transactions processed through fraud detection system';
COMMENT ON COLUMN transactions.velocity_5min IS 'Number of transactions in 5-minute window (from Kafka Streams velocity KTable)';

-- ═══════════════════════════════════════════════════════════════════════════
-- Table 2: customers
-- Customer profiles with persistent storage and versioning
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    avg_transaction_amount DECIMAL(12, 2),
    typical_merchant_category VARCHAR(50),
    account_age_days INTEGER CHECK (account_age_days >= 0),
    risk_score DECIMAL(5, 2) CHECK (risk_score BETWEEN 0 AND 100),
    total_transactions INTEGER DEFAULT 0,
    total_fraud_cases INTEGER DEFAULT 0,
    last_transaction_at TIMESTAMP,
    last_updated TIMESTAMP DEFAULT NOW(),
    version INTEGER DEFAULT 1,  -- Optimistic locking for concurrent updates

    CONSTRAINT chk_fraud_ratio CHECK (total_fraud_cases <= total_transactions)
);

CREATE INDEX idx_customers_risk_score ON customers(risk_score DESC);
CREATE INDEX idx_customers_last_updated ON customers(last_updated DESC);

COMMENT ON TABLE customers IS 'Persistent customer profiles (dual-write with Kafka KTable for fault tolerance)';
COMMENT ON COLUMN customers.version IS 'Optimistic locking version - increment on each update to detect concurrent modifications';

-- ═══════════════════════════════════════════════════════════════════════════
-- Table 3: fraud_decisions
-- Complete fraud decision audit trail with ML + LLM context
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE fraud_decisions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(50) NOT NULL REFERENCES transactions(transaction_id),

    -- ML layer outputs
    ml_score DECIMAL(5, 4) CHECK (ml_score BETWEEN 0 AND 1),
    ml_model_type VARCHAR(20) DEFAULT 'xgboost',
    shap_top_features JSONB,  -- Top 3 SHAP feature contributions

    -- LLM ensemble outputs
    llm_confidence DECIMAL(5, 2) CHECK (llm_confidence BETWEEN 0 AND 100),
    fraud_pattern VARCHAR(50),  -- card_testing, vpn_bot_fraud, account_takeover, etc.
    reasoning TEXT,
    recommendation TEXT,

    -- RAG context metadata
    rag_context_used BOOLEAN DEFAULT FALSE,
    rag_similarity_score DECIMAL(5, 4),
    rag_cases_retrieved INTEGER DEFAULT 0,

    -- Final routing decision
    decision VARCHAR(20) CHECK (decision IN ('BLOCK', 'REVIEW', 'APPROVE')),
    decided_at TIMESTAMP DEFAULT NOW(),

    -- Relaxed constraint: allows NULL similarity when RAG context is used
    -- (Original constraint was too strict - maxSimilarity not always available)
    CONSTRAINT chk_rag_similarity CHECK (
        rag_similarity_score IS NULL OR rag_context_used = TRUE
    )
);

CREATE INDEX idx_fraud_decisions_txn ON fraud_decisions(transaction_id);
CREATE INDEX idx_fraud_decisions_pattern ON fraud_decisions(fraud_pattern);
CREATE INDEX idx_fraud_decisions_time ON fraud_decisions(decided_at DESC);
CREATE INDEX idx_fraud_decisions_decision ON fraud_decisions(decision);
CREATE INDEX idx_fraud_decisions_ml_score ON fraud_decisions(ml_score DESC);

-- GIN index for JSONB queries (e.g., "which decisions had high SHAP values for velocity features?")
CREATE INDEX idx_fraud_decisions_shap ON fraud_decisions USING GIN(shap_top_features);

COMMENT ON TABLE fraud_decisions IS 'Immutable audit trail of every fraud decision (compliance requirement, 7+ year retention)';
COMMENT ON COLUMN fraud_decisions.fraud_pattern IS 'LLM-detected pattern from STREAMING_CONSENSUS_COORDINATOR agent';
COMMENT ON COLUMN fraud_decisions.shap_top_features IS 'JSONB array of {feature: string, value: float} from XGBoost TreeExplainer';

-- ═══════════════════════════════════════════════════════════════════════════
-- Table 4: case_reviews
-- Analyst workflow tracking and investigation notes
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE case_reviews (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(50) NOT NULL REFERENCES transactions(transaction_id),
    analyst_id VARCHAR(50),
    review_outcome VARCHAR(30) CHECK (review_outcome IN (
        'CONFIRMED_FRAUD',
        'FALSE_POSITIVE',
        'ESCALATED',
        'LEGITIMATE'
    )),
    notes TEXT,
    time_to_review_seconds INTEGER CHECK (time_to_review_seconds >= 0),
    reviewed_at TIMESTAMP DEFAULT NOW(),

    -- Link to original fraud decision
    fraud_decision_id INTEGER REFERENCES fraud_decisions(id),

    -- Investigation metadata
    evidence_sources JSONB,  -- Links to external systems, screenshots, customer contact logs
    confidence_level VARCHAR(20) CHECK (confidence_level IN ('LOW', 'MEDIUM', 'HIGH'))
);

CREATE INDEX idx_case_reviews_txn ON case_reviews(transaction_id);
CREATE INDEX idx_case_reviews_analyst ON case_reviews(analyst_id);
CREATE INDEX idx_case_reviews_outcome ON case_reviews(review_outcome);
CREATE INDEX idx_case_reviews_time ON case_reviews(reviewed_at DESC);

COMMENT ON TABLE case_reviews IS 'Analyst feedback loop - feeds ChromaDB embeddings and River online learning';
COMMENT ON COLUMN case_reviews.time_to_review_seconds IS 'SLA tracking: time from decision to analyst resolution';

-- ═══════════════════════════════════════════════════════════════════════════
-- Table 5: model_predictions
-- Model performance tracking and drift detection
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE model_predictions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(50) NOT NULL,
    model_type VARCHAR(30) CHECK (model_type IN (
        'xgboost',
        'river_srp',
        'llm_ensemble'
    )),
    prediction_score DECIMAL(5, 4) CHECK (prediction_score BETWEEN 0 AND 1),
    top_features JSONB,  -- Model-specific feature importances
    prediction_time_ms INTEGER CHECK (prediction_time_ms >= 0),
    predicted_at TIMESTAMP DEFAULT NOW(),

    -- Version tracking for A/B testing and rollbacks
    model_version VARCHAR(20),

    -- Ground truth comparison (populated after analyst feedback)
    actual_label BOOLEAN,  -- TRUE = fraud, FALSE = legitimate
    prediction_correct BOOLEAN,  -- Populated after case review
    updated_at TIMESTAMP
);

CREATE INDEX idx_model_predictions_txn ON model_predictions(transaction_id);
CREATE INDEX idx_model_predictions_type ON model_predictions(model_type);
CREATE INDEX idx_model_predictions_time ON model_predictions(predicted_at DESC);
CREATE INDEX idx_model_predictions_score ON model_predictions(prediction_score DESC);

COMMENT ON TABLE model_predictions IS 'Per-model prediction tracking for precision/recall analysis and model comparison';
COMMENT ON COLUMN model_predictions.prediction_time_ms IS 'Latency SLA tracking (target: <100ms for XGBoost, <2000ms for LLM ensemble)';

-- ═══════════════════════════════════════════════════════════════════════════
-- Table 6: feature_stats
-- Feature drift monitoring and data quality metrics
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE feature_stats (
    id SERIAL PRIMARY KEY,
    feature_name VARCHAR(100) NOT NULL,
    stat_type VARCHAR(20) CHECK (stat_type IN (
        'mean', 'stddev', 'median',
        'p25', 'p50', 'p75', 'p95', 'p99',
        'min', 'max', 'null_rate'
    )),
    stat_value DECIMAL(12, 4),
    sample_count INTEGER,
    window_start TIMESTAMP NOT NULL,
    window_end TIMESTAMP NOT NULL,
    computed_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT chk_window CHECK (window_end > window_start),
    CONSTRAINT uq_feature_stat_window UNIQUE (feature_name, stat_type, window_start)
);

CREATE INDEX idx_feature_stats_name_time ON feature_stats(feature_name, window_start DESC);
CREATE INDEX idx_feature_stats_type ON feature_stats(stat_type);

COMMENT ON TABLE feature_stats IS 'Feature drift detection - compare production stats vs training distribution';
COMMENT ON COLUMN feature_stats.null_rate IS 'Data quality metric: percentage of null/missing values in time window';

-- ═══════════════════════════════════════════════════════════════════════════
-- Analytics Views (for common queries)
-- ═══════════════════════════════════════════════════════════════════════════

-- View: fraud_summary_by_pattern
-- Aggregate fraud decisions by pattern with statistics
CREATE VIEW fraud_summary_by_pattern AS
SELECT
    fraud_pattern,
    COUNT(*) as total_cases,
    AVG(ml_score) as avg_ml_score,
    AVG(llm_confidence) as avg_llm_confidence,
    SUM(CASE WHEN decision = 'BLOCK' THEN 1 ELSE 0 END) as blocked_count,
    SUM(CASE WHEN decision = 'REVIEW' THEN 1 ELSE 0 END) as review_count,
    SUM(CASE WHEN rag_context_used THEN 1 ELSE 0 END) as rag_assisted_count,
    MIN(decided_at) as first_seen,
    MAX(decided_at) as last_seen
FROM fraud_decisions
WHERE fraud_pattern IS NOT NULL
GROUP BY fraud_pattern
ORDER BY total_cases DESC;

COMMENT ON VIEW fraud_summary_by_pattern IS 'Pattern-based fraud analytics for dashboard visualization';

-- View: analyst_performance
-- Track analyst review performance metrics
CREATE VIEW analyst_performance AS
SELECT
    analyst_id,
    COUNT(*) as total_reviews,
    AVG(time_to_review_seconds) as avg_review_time_sec,
    SUM(CASE WHEN review_outcome = 'CONFIRMED_FRAUD' THEN 1 ELSE 0 END) as confirmed_fraud,
    SUM(CASE WHEN review_outcome = 'FALSE_POSITIVE' THEN 1 ELSE 0 END) as false_positives,
    SUM(CASE WHEN review_outcome = 'ESCALATED' THEN 1 ELSE 0 END) as escalated,
    MIN(reviewed_at) as first_review,
    MAX(reviewed_at) as last_review
FROM case_reviews
WHERE analyst_id IS NOT NULL
GROUP BY analyst_id
ORDER BY total_reviews DESC;

COMMENT ON VIEW analyst_performance IS 'Analyst productivity and decision distribution metrics';

-- View: model_performance_comparison
-- Compare prediction accuracy across model types
CREATE VIEW model_performance_comparison AS
SELECT
    model_type,
    COUNT(*) as total_predictions,
    AVG(prediction_score) as avg_score,
    AVG(prediction_time_ms) as avg_latency_ms,
    SUM(CASE WHEN prediction_correct = TRUE THEN 1 ELSE 0 END)::FLOAT /
        NULLIF(SUM(CASE WHEN prediction_correct IS NOT NULL THEN 1 ELSE 0 END), 0) as accuracy,
    MIN(predicted_at) as first_prediction,
    MAX(predicted_at) as last_prediction
FROM model_predictions
GROUP BY model_type
ORDER BY model_type;

COMMENT ON VIEW model_performance_comparison IS 'Model accuracy and latency comparison (requires analyst feedback for ground truth)';

-- ═══════════════════════════════════════════════════════════════════════════
-- Sample Data (for testing)
-- ═══════════════════════════════════════════════════════════════════════════

-- Insert sample customer (for testing)
INSERT INTO customers (customer_id, avg_transaction_amount, typical_merchant_category, account_age_days, risk_score, total_transactions)
VALUES ('CUST-SAMPLE-001', 125.50, 'retail', 365, 25.0, 42)
ON CONFLICT (customer_id) DO NOTHING;

COMMENT ON SCHEMA public IS 'Fraud Detection System - Audit Trail & Analytics Layer';

-- Grant permissions (Docker Compose uses fraud_user by default)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO fraud_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO fraud_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO fraud_user;
