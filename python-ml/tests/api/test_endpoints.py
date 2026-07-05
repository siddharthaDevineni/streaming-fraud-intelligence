"""
Integration tests for the FastAPI sync endpoints.

Prerequisites:
  - FastAPI server running: python -m api.server
  - ChromaDB populated (run scenario 2 at least once first)

Run with:
  cd python-ml
  pytest tests/api/test_endpoints.py -v

Or against a different host:
  pytest tests/api/test_endpoints.py -v --base-url=http://localhost:8000
"""

import pytest
import httpx

BASE_URL = "http://localhost:8000"
ANALYZE_TIMEOUT = 60.0  # 10 LLM calls take ~2-3s each
CHAT_TIMEOUT = 30.0


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client() -> httpx.Client:
    """Shared httpx client for all endpoint tests."""
    with httpx.Client(base_url=BASE_URL, timeout=ANALYZE_TIMEOUT) as c:
        yield c


@pytest.fixture
def fraud_payload() -> dict:
    """High velocity card testing attack — expects isFraudulent=True."""
    return {
        "transaction": {
            "transactionId": "TEST-REST-001",
            "customerId": "CUST-001",
            "amount": 30,
            "currency": "USD",
            "merchantId": "MERCHANT-SUSPICIOUS-7",
            "merchantCategory": "ONLINE",
            "location": "Unknown Location",
            "timestamp": "2026-07-04T13:00:00",
            "metadata": {
                "deviceId": "BOT-DEVICE-1",
                "channel": "ONLINE",
                "rapidFire": True,
            },
        },
        "customerProfile": {
            "customerId": "CUST-001",
            "averageTransactionAmount": 244.0,
            "dailySpendingLimit": 1000.0,
            "transactionCategories": ["GROCERY", "ONLINE"],
            "primaryLocation": "Houston",
            "riskLevel": "HIGH",
        },
        "velocityCount": 15,
    }


@pytest.fixture
def legitimate_payload() -> dict:
    """Normal in-store grocery transaction — expects isFraudulent=False."""
    return {
        "transaction": {
            "transactionId": "TEST-REST-002",
            "customerId": "CUST-002",
            "amount": 48,
            "currency": "USD",
            "merchantId": "GROCERY-STORE-1",
            "merchantCategory": "GROCERY",
            "location": "Houston",
            "timestamp": "2026-07-04T13:00:00",
            "metadata": {
                "deviceId": "DEVICE-NORMAL-1",
                "channel": "IN_STORE",
                "rapidFire": False,
            },
        },
        "customerProfile": {
            "customerId": "CUST-002",
            "averageTransactionAmount": 178.0,
            "dailySpendingLimit": 500.0,
            "transactionCategories": ["GROCERY", "RESTAURANT"],
            "primaryLocation": "Houston",
            "riskLevel": "LOW",
        },
        "velocityCount": 1,
    }


@pytest.fixture
def no_profile_payload() -> dict:
    """Suspicious transaction without customer profile — agents still run."""
    return {
        "transaction": {
            "transactionId": "TEST-REST-003",
            "customerId": "CUST-UNKNOWN",
            "amount": 500,
            "currency": "USD",
            "merchantId": "MERCHANT-SUSPICIOUS-1",
            "merchantCategory": "ONLINE",
            "location": "Unknown Location",
            "timestamp": "2026-07-04T13:00:00",
            "metadata": {
                "deviceId": "BOT-DEVICE-2",
                "channel": "ONLINE",
                "rapidFire": True,
            },
        },
        "velocityCount": 8,
    }


# ── Health check ──────────────────────────────────────────────────────────────

class TestHealth:

    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert "timestamp" in data
        assert "services" in data

    def test_health_services_ready(self, client):
        services = client.get("/health").json()["services"]
        assert services["agent_coordinator"] == "ready"
        assert services["rag_retriever"] == "ready"
        assert services["chat_llm"] == "ready"


# ── /analyze endpoint ─────────────────────────────────────────────────────────

class TestAnalyze:

    def test_fraud_transaction_returns_200(self, client, fraud_payload):
        response = client.post("/analyze", json=fraud_payload)
        assert response.status_code == 200

    def test_fraud_transaction_detected(self, client, fraud_payload):
        data = client.post("/analyze", json=fraud_payload).json()
        assert data["isFraudulent"] is True, (
            f"Expected fraud detection, got isFraudulent=False "
            f"(confidence={data.get('confidenceScore')})"
        )

    def test_fraud_transaction_high_confidence(self, client, fraud_payload):
        data = client.post("/analyze", json=fraud_payload).json()
        assert data["confidenceScore"] >= 0.7, (
            f"Low confidence for clear fraud: {data['confidenceScore']}"
        )

    def test_fraud_transaction_uses_ten_agents(self, client, fraud_payload):
        """High velocity + customer profile → all collaboration rounds → 10 agents."""
        data = client.post("/analyze", json=fraud_payload).json()
        assert data["agentCount"] == 10, (
            f"Expected 10 agents, got {data['agentCount']}"
        )

    def test_fraud_response_structure(self, client, fraud_payload):
        data = client.post("/analyze", json=fraud_payload).json()
        assert "transactionId" in data
        assert "isFraudulent" in data
        assert "confidenceScore" in data
        assert "finalRiskScore" in data
        assert "agentCount" in data
        assert "explanation" in data
        assert "agentInsights" in data

    def test_fraud_agent_insights_count(self, client, fraud_payload):
        data = client.post("/analyze", json=fraud_payload).json()
        assert len(data["agentInsights"]) == 10

    def test_legitimate_transaction_returns_200(self, client, legitimate_payload):
        response = client.post("/analyze", json=legitimate_payload)
        assert response.status_code == 200

    def test_legitimate_transaction_not_flagged(self, client, legitimate_payload):
        data = client.post("/analyze", json=legitimate_payload).json()
        assert data["isFraudulent"] is False, (
            f"Expected legitimate, got isFraudulent=True "
            f"(confidence={data.get('confidenceScore')})"
        )

    def test_no_profile_transaction_returns_200(self, client, no_profile_payload):
        response = client.post("/analyze", json=no_profile_payload)
        assert response.status_code == 200

    def test_no_profile_transaction_returns_decision(self, client, no_profile_payload):
        """Agents should still produce a decision without a customer profile."""
        data = client.post("/analyze", json=no_profile_payload).json()
        assert "isFraudulent" in data
        assert "confidenceScore" in data
        assert 0.0 <= data["confidenceScore"] <= 1.0

    def test_no_profile_uses_fewer_agents(self, client, no_profile_payload):
        """No profile → profile collaboration skipped → 6 agents (5+0+1)."""
        data = client.post("/analyze", json=no_profile_payload).json()
        assert data["agentCount"] in (6, 8), (
            f"Expected 6 (no collab) or 8 (profile collab only), "
            f"got {data['agentCount']}"
        )

    def test_missing_required_field_returns_422(self, client):
        """Transaction missing currency → FastAPI validation error."""
        payload = {
            "transaction": {
                "transactionId": "TEST-INVALID",
                "customerId": "CUST-001",
                "amount": 30,
                # currency intentionally missing
                "merchantId": "MERCHANT-1",
                "merchantCategory": "ONLINE",
                "location": "Houston",
                "timestamp": "2026-07-04T13:00:00",
                "metadata": {},
            }
        }
        response = client.post("/analyze", json=payload)
        assert response.status_code == 422


# ── /investigation/chat endpoint ──────────────────────────────────────────────

class TestChat:

    def test_chat_returns_200(self, client):
        response = client.post(
            "/investigation/chat",
            json={"question": "How does the system detect fraud?"},
            timeout=CHAT_TIMEOUT,
        )
        assert response.status_code == 200

    def test_chat_response_structure(self, client):
        data = client.post(
            "/investigation/chat",
            json={"question": "How does the system detect fraud?"},
            timeout=CHAT_TIMEOUT,
        ).json()
        assert "response" in data
        assert "timestamp" in data
        assert "systemCapabilities" in data

    def test_chat_response_is_non_empty(self, client):
        data = client.post(
            "/investigation/chat",
            json={"question": "How does the system detect fraud?"},
            timeout=CHAT_TIMEOUT,
        ).json()
        assert len(data["response"]) > 100, "Response too short — likely an error"

    def test_chat_agent_collaboration_question(self, client):
        """Asks specifically about the parallel agent architecture."""
        data = client.post(
            "/investigation/chat",
            json={"question": "How do the agents collaborate to detect card testing attacks?"},
            timeout=CHAT_TIMEOUT,
        ).json()
        response_text = data["response"].lower()
        # Response should mention parallel execution and the three phases
        assert any(
            keyword in response_text
            for keyword in ["parallel", "phase 1", "simultaneously", "threadpool"]
        ), "Response did not mention parallel agent execution"

    def test_chat_rag_question(self, client):
        """Asks specifically about the ChromaDB RAG pipeline."""
        data = client.post(
            "/investigation/chat",
            json={
                "question": (
                    "What is the role of ChromaDB RAG in improving "
                    "fraud detection confidence?"
                )
            },
            timeout=CHAT_TIMEOUT,
        ).json()
        response_text = data["response"].lower()
        assert any(
            keyword in response_text
            for keyword in ["chromadb", "rag", "retrieval", "historical", "similar"]
        ), "Response did not mention ChromaDB or RAG"

    def test_chat_missing_question_returns_422(self, client):
        """Empty body → FastAPI validation error."""
        response = client.post(
            "/investigation/chat",
            json={},
            timeout=CHAT_TIMEOUT,
        )
        assert response.status_code == 422