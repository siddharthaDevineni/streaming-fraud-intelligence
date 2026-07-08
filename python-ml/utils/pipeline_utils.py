from agents.base_agent import AgentInsight
from config import settings
from models.schemas import EnrichedTransaction, CustomerProfile


def build_streaming_context(
        enriched: EnrichedTransaction,
        fraud_score: float | None = None,
        rag_context_text: str = ""
) -> str:
    """
    Builds the streaming context string injected into every agent prompt.
    Mirrors FraudStreams.java StreamingContext.getAIContext().

    Used by both:
    - inference_consumer.py (Kafka pipeline, has XGBoost fraud_score)
    - api/server.py (REST API, no fraud_score available)
    """
    parts = []

    if enriched.velocityCount and enriched.velocityCount >= settings.high_velocity_threshold:
        parts.append(
            f"HIGH VELOCITY: {enriched.velocityCount} transactions "
            f"in the last 5 minutes"
        )

    if enriched.customerProfile:
        profile: CustomerProfile = enriched.customerProfile
        parts.append(
            f"Customer baseline: ${profile.averageTransactionAmount:.2f} avg, "
            f"{profile.riskLevel} risk."
        )

    if fraud_score is not None:
        fraud_pct = round(fraud_score * 100, 1)
        if fraud_score >= 0.7:
            parts.append(
                f"ML MODEL PRE-SCREEN: XGBoost fraud score = {fraud_pct}%. "
                f"ML strongly indicates fraud."
            )
        elif fraud_score >= 0.3:
            parts.append(
                f"ML MODEL PRE-SCREEN: XGBoost fraud score = {fraud_pct}%. "
                f"ML indicates moderate risk."
            )
        else:
            parts.append(
                f"ML MODEL PRE-SCREEN: XGBoost fraud score = {fraud_pct}%. "
                f"ML indicates likely legitimate."
            )

    if rag_context_text:
        parts.append(rag_context_text)

    return "\n".join(parts) if parts else "No additional streaming context available."


def enriched_to_dict(
        enriched: EnrichedTransaction,
        features: dict,
) -> dict:
    """
    Converts EnrichedTransaction to the dict format AgentCoordinator expects.

    Used by both:
    - inference_consumer.py (Kafka pipeline)
    - api/server.py (REST API)
    """
    txn = enriched.transaction
    profile: CustomerProfile | None = enriched.customerProfile

    return {
        "transactionId": txn.transactionId,
        "customerId": txn.customerId,
        "amount": txn.amount,
        "currency": getattr(txn, "currency", "USD"),
        "merchantId": txn.merchantId,
        "merchantCategory": getattr(txn, "merchantCategory", "UNKNOWN"),
        "location": getattr(txn, "location", "Unknown"),
        "metadata": getattr(txn, "metadata", {}),
        "velocityCount": enriched.velocityCount or 0,
        "hasHighVelocity": bool(features.get("is_high_velocity", 0)),
        "isAmountUnusual": bool(features.get("is_amount_unusual", 0)),
        "customerRiskLevel": profile.riskLevel if profile else "UNKNOWN",
        "customerAvgAmount": profile.averageTransactionAmount if profile else 0,
    }