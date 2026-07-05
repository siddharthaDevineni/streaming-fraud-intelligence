"""
Unit tests for Python LLM agent layer.

Run with:
  cd python-ml
  pytest tests/agents/test_agents.py -v

Each agent is tested individually, then AgentCoordinator is tested
for the full 3-phase pipeline (10 LLM calls per transaction).
"""

import pytest
from agents.agent_coordinator import AgentCoordinator
from agents.base_agent import AgentInsight
from agents.behavior_analyst import BehaviorAnalyst
from agents.geographic_analyst import GeographicAnalyst
from agents.pattern_detector import PatternDetector
from agents.risk_assessor import RiskAssessor
from agents.temporal_analyst import TemporalAnalyst


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fraud_transaction() -> dict:
    """High velocity card testing attack — clear fraud signals."""
    return {
        "transactionId": "TEST-001",
        "amount": 30,
        "currency": "USD",
        "merchantId": "MERCHANT-SUSPICIOUS-1",
        "merchantCategory": "ONLINE",
        "location": "Unknown Location",
        "metadata": {
            "deviceId": "BOT-DEVICE-1",
            "channel": "ONLINE",
            "rapidFire": True,
        },
    }


@pytest.fixture(scope="module")
def fraud_transaction_with_context() -> dict:
    """Fraud transaction with full coordinator context fields."""
    return {
        "transactionId": "TEST-001",
        "amount": 30,
        "currency": "USD",
        "merchantId": "MERCHANT-SUSPICIOUS-1",
        "merchantCategory": "ONLINE",
        "location": "Unknown Location",
        "velocityCount": 15,
        "hasHighVelocity": True,
        "isAmountUnusual": True,
        "customerRiskLevel": "HIGH",
        "customerAvgAmount": 244,
        "metadata": {
            "deviceId": "BOT-DEVICE-1",
            "channel": "ONLINE",
            "rapidFire": True,
        },
    }


@pytest.fixture(scope="module")
def streaming_context_simple() -> str:
    return (
        "HIGH VELOCITY: 15 transactions in 5 minutes. "
        "Customer baseline: $244 avg, HIGH risk. "
        "XGBoost fraud score = 99.8%."
    )


@pytest.fixture(scope="module")
def streaming_context_full() -> str:
    return (
        "HIGH VELOCITY: 15 transactions in 5 minutes\n"
        "Customer baseline: $244 avg, HIGH risk.\n"
        "XGBoost fraud score = 99.8%. ML strongly indicates fraud.\n"
        "3 similar confirmed cases retrieved at 82% similarity — all card_testing."
    )


@pytest.fixture(scope="module")
def coordinator() -> AgentCoordinator:
    return AgentCoordinator()


# ── Individual agent tests ────────────────────────────────────────────────────

class TestIndividualAgents:
    """Each agent produces an AgentInsight with a valid risk score."""

    @pytest.mark.parametrize("agent_class,expected_name,expected_weight", [
        (BehaviorAnalyst,   "BEHAVIOR_ANALYST",   1.2),
        (PatternDetector,   "PATTERN_DETECTOR",   1.3),
        (RiskAssessor,      "RISK_ASSESSOR",       1.1),
        (GeographicAnalyst, "GEOGRAPHIC_ANALYST",  1.0),
        (TemporalAnalyst,   "TEMPORAL_ANALYST",    1.0),
    ])
    def test_agent_properties(self, agent_class, expected_name, expected_weight):
        agent = agent_class()
        assert agent.agent_name == expected_name
        assert agent.weight == expected_weight

    @pytest.mark.parametrize("agent_class", [
        BehaviorAnalyst,
        PatternDetector,
        RiskAssessor,
        GeographicAnalyst,
        TemporalAnalyst,
    ])
    def test_agent_analyze_returns_valid_insight(
            self,
            agent_class,
            fraud_transaction,
            streaming_context_simple,
    ):
        agent = agent_class()
        insight = agent.analyze(fraud_transaction, streaming_context_simple)

        assert isinstance(insight, AgentInsight)
        assert 0.0 <= insight.risk_score <= 1.0
        assert 0.0 <= insight.confidence <= 1.0
        assert insight.agent_name == agent.agent_name
        assert insight.weight == agent.weight
        assert len(insight.reasoning) > 0

    @pytest.mark.parametrize("agent_class", [
        BehaviorAnalyst,
        PatternDetector,
        RiskAssessor,
        GeographicAnalyst,
        TemporalAnalyst,
    ])
    def test_agent_flags_fraud_for_suspicious_transaction(
            self,
            agent_class,
            fraud_transaction,
            streaming_context_simple,
    ):
        """All agents should score >= 0.6 for a clear fraud transaction."""
        agent = agent_class()
        insight = agent.analyze(fraud_transaction, streaming_context_simple)

        assert insight.risk_score >= 0.6, (
            f"{agent.agent_name} scored only {insight.risk_score} "
            f"for an obvious fraud transaction"
        )
        assert insight.indicates_fraud(), (
            f"{agent.agent_name} did not flag fraud despite risk={insight.risk_score}"
        )


# ── AgentCoordinator tests ────────────────────────────────────────────────────

class TestAgentCoordinator:
    """Full 3-phase pipeline producing a FraudDecision."""

    def test_coordinator_initialises_five_agents(self, coordinator):
        assert len(coordinator.agents) == 5

    def test_coordinator_produces_fraud_decision_structure(
            self,
            coordinator,
            fraud_transaction_with_context,
            streaming_context_full,
    ):
        decision = coordinator.investigate(
            transaction=fraud_transaction_with_context,
            streaming_context=streaming_context_full,
            has_high_velocity=True,
            has_customer_profile=True,
        )

        assert "isFraudulent" in decision
        assert "confidenceScore" in decision
        assert "finalRiskScore" in decision
        assert "agentCount" in decision
        assert "explanation" in decision
        assert "agentInsights" in decision

    def test_coordinator_detects_fraud(
            self,
            coordinator,
            fraud_transaction_with_context,
            streaming_context_full,
    ):
        decision = coordinator.investigate(
            transaction=fraud_transaction_with_context,
            streaming_context=streaming_context_full,
            has_high_velocity=True,
            has_customer_profile=True,
        )

        assert decision["isFraudulent"] is True, (
            f"Expected fraud, got isFraudulent=False "
            f"(confidence={decision['confidenceScore']})"
        )

    def test_coordinator_uses_ten_agents_with_full_collaboration(
            self,
            coordinator,
            fraud_transaction_with_context,
            streaming_context_full,
    ):
        """High velocity + customer profile → all collaboration rounds fire → 10 agents."""
        decision = coordinator.investigate(
            transaction=fraud_transaction_with_context,
            streaming_context=streaming_context_full,
            has_high_velocity=True,
            has_customer_profile=True,
        )

        assert decision["agentCount"] == 10, (
            f"Expected 10 agents (5 parallel + 4 collaboration + 1 consensus), "
            f"got {decision['agentCount']}"
        )

    def test_coordinator_high_confidence_for_clear_fraud(
            self,
            coordinator,
            fraud_transaction_with_context,
            streaming_context_full,
    ):
        decision = coordinator.investigate(
            transaction=fraud_transaction_with_context,
            streaming_context=streaming_context_full,
            has_high_velocity=True,
            has_customer_profile=True,
        )

        assert decision["confidenceScore"] >= 0.7, (
            f"Expected confidence >= 0.7, got {decision['confidenceScore']}"
        )

    def test_coordinator_fewer_agents_without_velocity(
            self,
            coordinator,
            fraud_transaction_with_context,
            streaming_context_full,
    ):
        """No high velocity → velocity collaboration skipped → 8 agents (5+2+1)."""
        decision = coordinator.investigate(
            transaction=fraud_transaction_with_context,
            streaming_context=streaming_context_full,
            has_high_velocity=False,
            has_customer_profile=True,
        )

        assert decision["agentCount"] == 8, (
            f"Expected 8 agents without velocity collaboration, "
            f"got {decision['agentCount']}"
        )

    def test_coordinator_agent_insights_populated(
            self,
            coordinator,
            fraud_transaction_with_context,
            streaming_context_full,
    ):
        decision = coordinator.investigate(
            transaction=fraud_transaction_with_context,
            streaming_context=streaming_context_full,
            has_high_velocity=True,
            has_customer_profile=True,
        )

        insights = decision["agentInsights"]
        assert len(insights) == 10
        for insight in insights:
            assert "agentName" in insight
            assert "riskScore" in insight
            assert 0.0 <= insight["riskScore"] <= 1.0