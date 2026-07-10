"""
Feedback Embedder
=================
Consumes: analyst-feedback (FraudDecision auto-sink from FraudStreams.java)
          ml-predictions (XGBoost inference output from inference_consumer.py)

Produces: ChromaDB collection "confirmed_fraud_cases"
          → consumed by rag_retriever.py at inference time

How it works:
  1. Subscribes to BOTH topics with a single consumer
  2. Caches ml-predictions by transactionId (features needed for embedding text)
  3. When analyst-feedback arrives with high confidence:
     → looks up cached features from ml-predictions
     → builds structured fraud case text
     → embeds with sentence-transformers (all-MiniLM-L6-v2, local, free)
     → stores in ChromaDB with metadata for filtering
"""

import json
import time
from datetime import datetime, timezone

import chromadb
import structlog
from config import settings
from confluent_kafka import Consumer, KafkaError
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger()

# ── Embedding thresholds ──────────────────────────────────────────────────────
# Only embed cases above this confidence — low confidence predictions
# are too uncertain to be useful as training examples for RAG
MIN_CONFIDENCE_TO_EMBED = 0.70

# How long to keep ml-predictions in memory cache (seconds)
# Feedback typically arrives within a few seconds of the prediction
CACHE_TTL_SECONDS = 300  # 5 minutes

# Maximum cache size — prevents unbounded memory growth
MAX_CACHE_SIZE = 1000


class FeedbackEmbedder:

    def __init__(self):
        self.consumer = self._create_consumer()
        self.embedding_model = self._load_embedding_model()
        self.collection = self._init_chromadb()

        # In-memory cache: transactionId → {features, mlScore, timestamp}
        # Used to join analyst-feedback with ml-predictions features
        self.prediction_cache: dict = {}
        self.pending_feedback: dict = {}
        self.embedded_count = 0

        logger.info(
            "feedback_embedder_ready",
            collection=settings.chroma_collection_fraud,
            existing_cases=self.collection.count(),
        )

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _create_consumer(self) -> Consumer:
        return Consumer({
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": "fraud-feedback-embedder",
            "auto.offset.reset": "latest",
            "enable.auto.commit": True,
        })

    def _load_embedding_model(self) -> SentenceTransformer:
        """
        Load sentence-transformers model locally.
        all-MiniLM-L6-v2: 384 dimensions, < 5ms per embedding on CPU.
        Downloads once (~80MB), cached in ~/.cache/huggingface.
        """
        logger.info("loading_embedding_model", model="all-MiniLM-L6-v2")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("embedding_model_ready")
        return model

    def _init_chromadb(self):
        """
        Initialize ChromaDB with persistent storage.
        Creates collection if it doesn't exist, reuses if it does.
        On app restart, existing cases are preserved and available immediately.
        """
        client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=chromadb.Settings(anonymized_telemetry=False))
        collection = client.get_or_create_collection(
            name=settings.chroma_collection_fraud,
            metadata={
                "hnsw:space": "cosine",  # cosine similarity for text
                "hnsw:construction_ef": 100,
                "hnsw:M": 16,
            },
        )
        logger.info(
            "chromadb_ready",
            path=settings.chroma_persist_dir,
            collection=settings.chroma_collection_fraud,
            existing_cases=collection.count(),
        )
        return collection

    # ── Text building ─────────────────────────────────────────────────────────

    def _build_fraud_case_text(
            self,
            features: dict,
            feedback: dict,
            ml_score: float,
            pattern: str
    ) -> str:
        """
        Build structured text representation of a confirmed fraud case.

        This text is what gets embedded. The format is deliberately
        structured and consistent so that:
        1. Similar fraud patterns produce similar vectors
        2. Retrieval matches on fraud SIGNATURE not narrative prose
        3. The injected context is readable by LLM agents

        Key design: use the same feature names as in features/engineer.py
        so the query text at retrieval time uses identical vocabulary.
        """
        velocity = features.get("velocity_count", 1)
        amount_ratio = features.get("amount_ratio", 1.0)
        is_high_velocity = features.get("is_high_velocity", 0)
        is_unknown_location = features.get("is_unknown_location", 0)
        is_suspicious_merchant = features.get("is_suspicious_merchant", 0)
        is_bot_device = features.get("is_bot_device", 0)
        is_rapid_fire = features.get("is_rapid_fire", 0)
        is_online = features.get("is_online", 0)
        is_typical_category = features.get("is_typical_category", 0)
        is_high_risk = features.get("is_high_risk_customer", 0)
        is_low_risk = features.get("is_low_risk_customer", 0)
        is_amount_unusual = features.get("is_amount_unusual", 0)
        hour = features.get("hour", 12)

        customer_risk = (
            "HIGH" if is_high_risk
            else "LOW" if is_low_risk
            else "MEDIUM"
        )

        merchant_type = "suspicious online" if (is_suspicious_merchant and is_online) \
            else "suspicious" if is_suspicious_merchant \
            else "online" if is_online \
            else "in-person"

        location_status = "unknown/vpn" if is_unknown_location else "known location"

        confidence = feedback.get("confidence", 0)
        agent_count = feedback.get("agentConsensus", 0)

        # actualFraud from human analyst if available, else system prediction
        confirmed_by = "human_analyst" if feedback.get("actualFraud") is not None \
            else "system_high_confidence"

        return f"""fraud pattern: {pattern}
            velocity: {velocity} transactions in 5 minutes {"(high velocity attack)" if is_high_velocity else ""}
            amount ratio: {amount_ratio:.2f}x customer baseline {"(unusual amount)" if is_amount_unusual else ""}
            merchant: {merchant_type} {"(atypical category)" if not is_typical_category else ""}
            location: {location_status}
            device: {"bot device detected" if is_bot_device else "normal device"}
            rapid fire flag: {"yes" if is_rapid_fire else "no"}
            transaction hour: {hour}:00
            customer risk tier: {customer_risk}
            ml fraud score: {ml_score:.3f}
            agent consensus: {agent_count} agents agreed
            decision confidence: {confidence:.2f}
            outcome: FRAUD CONFIRMED
            confirmed by: {confirmed_by}"""

    # ── Embedding and storage ─────────────────────────────────────────────────

    def _embed_and_store(
            self,
            transaction_id: str,
            customer_id: str,
            features: dict,
            feedback: dict,
            ml_score: float,
            fraud_pattern: str = "unknown"
    ) -> None:
        """
        Embed the fraud case text and store in ChromaDB.

        Metadata stored alongside the vector:
        - fraudPattern: LLM-assessed pattern type (card_testing, vpn_bot_fraud etc.)
        - customerId:   for customer-specific retrieval priority
        - isHighVelocity: kept for informational purposes (no longer used as filter)
        - confidence:   agent consensus confidence at time of confirmation
        - Shown to agents as context alongside the retrieved text
        - Not used for similarity matching (that uses the vector only)
        """
        pattern = fraud_pattern
        fraud_case_text = self._build_fraud_case_text(features, feedback, ml_score, pattern)

        # Check if already embedded (idempotent — safe to run multiple times)
        existing = self.collection.get(ids=[transaction_id])
        if existing["ids"]:
            logger.debug(
                "case_already_embedded",
                transaction_id=transaction_id,
            )
            return

        # Embed: text → 384-dimensional vector
        embedding = self.embedding_model.encode(fraud_case_text).tolist()

        # Store in ChromaDB
        self.collection.add(
            ids=[transaction_id],
            documents=[fraud_case_text],
            embeddings=[embedding],
            metadatas=[{
                "transactionId": transaction_id,
                "customerId": customer_id,
                "fraudPattern": pattern,
                "confidence": float(feedback.get("confidence", 0)),
                "mlFraudScore": float(ml_score),
                "velocityCount": int(features.get("velocity_count", 1)),
                "isHighVelocity": int(features.get("is_high_velocity", 0)),
                "isBotDevice": int(features.get("is_bot_device", 0)),
                "isSuspiciousMerchant": int(features.get("is_suspicious_merchant", 0)),
                "confirmedBy": "human_analyst"
                if feedback.get("actualFraud") is not None
                else "system",
                "embeddedAt": datetime.now(timezone.utc).isoformat(),
            }]
        )

        self.embedded_count += 1
        logger.info(
            "fraud_case_embedded",
            transaction_id=transaction_id,
            pattern=pattern,
            confidence=feedback.get("confidence"),
            total_cases=self.collection.count(),
        )

    # ── Cache management ──────────────────────────────────────────────────────

    def _cache_prediction(self, prediction: dict):
        """Cache ML prediction features for later join with feedback."""
        txn_id = prediction.get("transactionId")
        if not txn_id:
            return

        self.prediction_cache[txn_id] = {
            "features": prediction.get("featuresUsed", {}),
            "mlFraudScore": prediction.get("mlFraudScore", 0.0),
            "customerId": prediction.get("customerId", "unknown"),
            "cachedAt": time.time(),
        }

        # Evict old entries to prevent unbounded growth
        if len(self.prediction_cache) > MAX_CACHE_SIZE:
            now = time.time()
            expired = [
                k for k, v in self.prediction_cache.items()
                if now - v["cachedAt"] > CACHE_TTL_SECONDS
            ]
            for k in expired:
                del self.prediction_cache[k]

            logger.debug(
                "cache_evicted",
                evicted=len(expired),
                remaining=len(self.prediction_cache),
            )

    def _should_embed(self, feedback: dict) -> bool:
        """
        Decide whether this feedback record is worth embedding.

        Embed if:
        - Human analyst explicitly confirmed fraud (actualFraud=True)
        - OR system predicted fraud with high confidence (> 0.80)

        Do NOT embed:
        - Low confidence predictions (too uncertain)
        - Legitimate transactions (not useful for fraud RAG)
        - System predictions without human confirmation below threshold
        """
        actual_fraud = feedback.get("actualFraud")
        predicted_fraud = feedback.get("predictedFraud", False)
        confidence = round(feedback.get("confidence", 0), 4)

        # Human confirmed fraud — always embed
        if actual_fraud is True:
            return True

        # Human confirmed NOT fraud — never embed (could seed false cases)
        if actual_fraud is False:
            return False

        # No human label yet — embed only if high-confidence system prediction
        return predicted_fraud and confidence >= MIN_CONFIDENCE_TO_EMBED

    # ── Message processing ────────────────────────────────────────────────────

    def _process_ml_prediction(self, raw: dict):
        """Cache ML prediction for later join with feedback."""
        txn_id = raw.get("transactionId")
        features = raw.get("featuresUsed", {})
        logger.info("ml_prediction_cached",
                    transaction_id=txn_id,
                    features_count=len(features),
                    ml_score=raw.get("mlFraudScore"))

        self._cache_prediction(raw)
        txn_id = raw.get("transactionId")

        # Check if analyst-feedback arrived before this ml-prediction
        if txn_id in self.pending_feedback:
            logger.info(
                "processing_pending_feedback",
                transaction_id=txn_id,
            )
            pending = self.pending_feedback.pop(txn_id)
            self._process_feedback(pending)  # retry now that cache has features

    def _process_feedback(self, raw: dict):
        """Process analyst feedback — embed if high confidence fraud."""
        txn_id = raw.get("transactionId")

        logger.info("feedback_received",
                    transaction_id=txn_id,
                    predicted_fraud=raw.get("predictedFraud"),
                    confidence=raw.get("confidence"),
                    should_embed=self._should_embed(raw),
                    in_cache=txn_id in self.prediction_cache)

        if not txn_id:
            return

        if not self._should_embed(raw):
            logger.debug(
                "feedback_skipped",
                transaction_id=txn_id,
                predicted_fraud=raw.get("predictedFraud"),
                confidence=raw.get("confidence"),
                actual_fraud=raw.get("actualFraud"),
            )
            return

        # Look up features from cache
        cached = self.prediction_cache.get(txn_id)
        if not cached:
            # ml-prediction not cached yet — store for later
            logger.debug(
                "feedback_pending",
                transaction_id=txn_id,
                hint="waiting for ml-predictions to be cached",
            )
            self.pending_feedback[txn_id] = raw  # ← store, don't discard
            return

        features = cached["features"]
        ml_score = cached["mlFraudScore"]
        velocity = features.get("velocity_count", 0)
        fraud_pattern = raw.get("fraudPattern", "unknown")

        # Skip embedding low-quality cold-start cases — transactions with
        # velocity=1 and near-zero ML score are timing lag artifacts, not
        # representative fraud patterns. They would poison ChromaDB with
        # vpn_bot_fraud cases that match every subsequent transaction.
        if velocity <= 2 and ml_score < 0.1:
            logger.debug(
                "embedding_skipped_cold_start_artifact",
                transaction_id=txn_id,
                velocity=velocity,
                ml_score=ml_score,
            )
            return

        self._embed_and_store(
            transaction_id=txn_id,
            customer_id=cached["customerId"],
            features=features,
            feedback=raw,
            ml_score=ml_score,
            fraud_pattern=fraud_pattern
        )

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        """
        Subscribe to both topics simultaneously.
        Single consumer handles both — messages from either topic
        are processed in arrival order.
        """
        self.consumer.subscribe([
            settings.topic_ml_predictions,  # cache features first
            settings.topic_analyst_feedback,  # then embed when feedback arrives
        ])

        logger.info(
            "feedback_embedder_consuming",
            topics=[settings.topic_ml_predictions, settings.topic_analyst_feedback],
        )

        try:
            while True:
                msg = self.consumer.poll(timeout=1.0)

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    logger.error("consumer_error", error=str(msg.error()))
                    continue

                try:
                    raw = json.loads(msg.value().decode("utf-8"))
                    topic = msg.topic()

                    if topic == settings.topic_ml_predictions:
                        self._process_ml_prediction(raw)
                    elif topic == settings.topic_analyst_feedback:
                        self._process_feedback(raw)

                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(
                        "message_processing_failed",
                        topic=msg.topic(),
                        error=str(e),
                    )

        except KeyboardInterrupt:
            logger.info(
                "shutting_down",
                total_embedded=self.embedded_count,
                total_cases_in_db=self.collection.count(),
            )
        finally:
            self.consumer.close()


if __name__ == "__main__":
    embedder = FeedbackEmbedder()
    embedder.run()
