"""
ML Inference Service (v2 — scaler fix applied)
===============================================
Consumes: enriched-transactions (produced by FraudStreams.java)
Produces: ml-predictions        (consumed by AgentCoordinator.java via StreamingContext)

This is the Python bridge between Kafka Streams (Java) and the ML model layer.
Every enriched transaction gets:
  1. Feature engineered (Polars)
  2. Scored by XGBoost (fraud probability 0.0-1.0)
  3. Explained by SHAP (top feature contributions)
  4. Published to ml-predictions topic
"""

import json

import joblib
from agents.agent_coordinator import AgentCoordinator
from dotenv import load_dotenv
from utils.pipeline_utils import build_streaming_context, enriched_to_dict

load_dotenv()
import numpy as np
import shap
import structlog
from langsmith import traceable
import time
from agents.rag_retriever import RAGRetriever
from config import settings
from confluent_kafka import Consumer, Producer, KafkaError
from datetime import datetime, timezone
from features.engineer import extract_features, FEATURE_COLUMNS
from models.schemas import EnrichedTransaction, MLPrediction, AgentInsightOutput, FraudDecisionOutput
from pydantic import ValidationError
from db.postgres_client import get_postgres_client

logger = structlog.get_logger()


class MLInferenceService:

    def __init__(self):
        self.rag_retriever = RAGRetriever()
        self.consumer = self._create_consumer()

        self.producer = self._create_producer()
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        self.explainer = shap.TreeExplainer(self.model)
        self.agent_coordinator = AgentCoordinator()

        # Initialize PostgreSQL client for audit trail persistence
        try:
            self.postgres_client = get_postgres_client(settings.postgres_url)
            logger.info("postgres_client_initialized")
        except Exception as e:
            logger.warning("postgres_client_init_failed", error=str(e),
                          hint="Postgres persistence disabled - will continue without database")
            self.postgres_client = None

        logger.info("ml_inference_service_started",
                    model_path=settings.xgboost_model_path)

    def _create_consumer(self) -> Consumer:
        return Consumer({
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": settings.consumer_group_inference,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        })

    def _create_producer(self) -> Producer:
        return Producer({
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "acks": "all",  # match Java EXACTLY_ONCE_V2 guarantee
        })

    def _load_model(self):
        """
        Load pre-trained XGBoost model.
        If the model doesn't exist yet, returns None — training must be run first.
        See training/train_xgboost.py
        """
        try:
            model = joblib.load(settings.xgboost_model_path)
            logger.info("model_loaded", path=settings.xgboost_model_path)
            return model
        except FileNotFoundError:
            logger.warning("model_not_found",
                           path=settings.xgboost_model_path,
                           hint="Run training/train_xgboost.py first")
            return None

    def _load_scaler(self):
        try:
            scaler = joblib.load(settings.scaler_path)
            logger.info("scaler_loaded", path=settings.scaler_path)
            return scaler
        except FileNotFoundError:
            logger.warning("scaler_not_found",
                           path=settings.scaler_path,
                           hint="Run training/train_xgboost.py first")
            return None

    @traceable(name="xgboost-inference-shap", run_type="chain")
    def _score(self, features: dict) -> tuple[float, dict]:
        """
        Run XGBoost inference and SHAP explanation.
        Returns (fraud_score, shap_explanation)
        """
        # Build feature vector in exact column order
        feature_vector = np.array([[features[col] for col in FEATURE_COLUMNS]])
        feature_vector_scaled = self.scaler.transform(feature_vector)

        # XGBoost inference — P(fraud)
        fraud_score = float(self.model.predict_proba(feature_vector_scaled)[0][1])

        # SHAP explanation — which features drove this score
        shap_values = self.explainer.shap_values(feature_vector_scaled)[0]
        shap_dict = dict(zip(FEATURE_COLUMNS, shap_values.tolist()))

        # Return only top 3 contributors for the Kafka message (keeps payload small)
        top_3 = dict(sorted(shap_dict.items(),
                            key=lambda x: abs(x[1]),
                            reverse=True)[:3])

        return fraud_score, top_3

    def _publish_prediction(self, prediction: MLPrediction):
        """Publish ML prediction to ml-predictions topic for Java to consume"""
        self.producer.produce(
            topic=settings.topic_ml_predictions,
            key=prediction.customerId,
            value=prediction.model_dump_json(),
            callback=self._delivery_callback,
        )
        self.producer.poll(0)

    def _delivery_callback(self, err, msg):
        if err:
            logger.error("prediction_delivery_failed", error=str(err))
        else:
            logger.debug("prediction_delivered",
                         topic=msg.topic(),
                         partition=msg.partition())

    def run(self):
        """Main consumption loop"""
        self.consumer.subscribe([settings.topic_enriched_transactions])
        logger.info("consuming", topic=settings.topic_enriched_transactions)

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

                self._process_message(msg)

        except KeyboardInterrupt:
            logger.info("shutting_down")
        finally:
            self.consumer.close()
            self.producer.flush()

    @traceable(name="chromadb-rag-retrieval", run_type="retriever")
    def _traced_rag_retrieve(self, features: dict, customer_id: str, n_results: int) -> list[dict]:
        """LangSmith-traced wrapper around RAG retrieval."""
        return self.rag_retriever.retrieve(
            features=features,
            customer_id=customer_id,
            n_results=n_results,
        )

    @traceable(name="fraud-ml-pipeline", run_type="chain")
    def _process_message(self, msg):
        start_ms = time.time() * 1000

        try:
            raw = json.loads(msg.value().decode("utf-8"))
            enriched = EnrichedTransaction(**raw)
        except (ValidationError, json.JSONDecodeError) as e:
            logger.error("deserialization_failed",
                         error=str(e),
                         raw_value=msg.value()[:200])
            return

        if self.model is None:
            logger.warning("model_not_loaded_skipping",
                           transaction_id=enriched.transaction.transactionId)
            return

        try:
            # Step 1 — feature engineering
            features = extract_features(enriched)

            # Step 2 — XGBoost inference + SHAP
            fraud_score, shap_top3 = self._score(features)

            # Step 3 — RAG retrieval (now feeds directly into agents — no KTable lag)
            similar_cases = self._traced_rag_retrieve(
                features=features,
                customer_id=enriched.transaction.customerId,
                n_results=3,
            )
            rag_context = self.rag_retriever.format_for_ml_prediction(similar_cases)
            rag_context_text = self.rag_retriever.format_for_agent_prompt(similar_cases)

            # Step 4 — build streaming context string for agents
            streaming_context = build_streaming_context(
                enriched=enriched,
                fraud_score=fraud_score,
                rag_context_text=rag_context_text,
            )

            # Step 5 — agent coordinator (10 LLM calls with full RAG context)
            transaction_dict = enriched_to_dict(enriched, features)
            fraud_decision = self.agent_coordinator.investigate(
                transaction=transaction_dict,
                streaming_context=streaming_context,
                has_high_velocity=bool(features.get("is_high_velocity", 0)),
                has_customer_profile=enriched.customerProfile is not None,
            )

            latency_ms = int(time.time() * 1000 - start_ms)

            # Step 6 — publish MLPrediction (Java reads via KTable for routing)
            prediction = MLPrediction(
                transactionId=enriched.transaction.transactionId,
                customerId=enriched.transaction.customerId,
                mlFraudScore=fraud_score,
                lstmSequenceScore=0.0,
                combinedScore=fraud_score,
                modelVersion="xgb-v1",
                shapExplanation=shap_top3,
                featuresUsed=features,
                ragContext=rag_context,
                inferenceLatencyMs=latency_ms,
                timestamp=datetime.now(timezone.utc),
            )

            # Publish MLPrediction — consumed by feedback_embedder.py (ChromaDB) and
            # the Streamlit dashboard for live monitoring. Java no longer reads this
            self._publish_prediction(prediction)

            # Publish FraudDecision (Java reads this for routing — replaces agentCoordinator)
            self._publish_decision(fraud_decision, prediction)

            # Persist to PostgreSQL for long-term audit trail (dual-write pattern)
            self._persist_to_postgres(enriched, prediction, fraud_decision, shap_top3, latency_ms)

            logger.info(
                "inference_complete",
                transaction_id=enriched.transaction.transactionId,
                fraud_score=round(fraud_score, 3),
                is_fraudulent=fraud_decision["isFraudulent"],
                confidence=round(fraud_decision["confidenceScore"], 3),
                agents=fraud_decision["agentCount"],
                latency_ms=latency_ms,
                top_shap_feature=list(shap_top3.keys())[0] if shap_top3 else None,
            )

        except Exception as e:
            logger.error("inference_failed",
                         transaction_id=enriched.transaction.transactionId,
                         error=str(e))

    def _publish_decision(self, fraud_decision: dict, prediction: MLPrediction):
        """
        Publish FraudDecision to fraud-decisions topic.
        Java reads this via KTable join and routes directly — no agent calls.
        Schema must match FraudDecision.java record exactly.
        """
        # Convert agent insights to Java-compatible format
        agent_insights = [
            AgentInsightOutput(
                agentType=insight.get("agentName", "UNKNOWN"),
                agentName=insight.get("agentName", "UNKNOWN"),
                analysis=insight.get("reasoning", ""),
                riskScore=insight.get("riskScore", 0.5),
                confidence=min(insight.get("riskScore", 0.5), 1.0),
                reasoning=insight.get("reasoning", ""),
                recommendation=self._risk_to_recommendation(
                    insight.get("riskScore", 0.5)
                ),
                timestamp=datetime.now().replace(tzinfo=None).isoformat(),
            )
            for insight in fraud_decision.get("agentInsights", [])
        ]

        output = FraudDecisionOutput(
            transactionId=fraud_decision["transactionId"],
            isFraudulent=fraud_decision["isFraudulent"],
            confidenceScore=fraud_decision["confidenceScore"],
            fraudPattern=fraud_decision.get("fraudPattern", "unknown"),
            primaryReason=(
                "AI agents with streaming intelligence detected fraud"
                if fraud_decision["isFraudulent"]
                else "Transaction appears legitimate"
            ),
            detailedExplanation=fraud_decision.get("explanation", ""),
            agentInsights=agent_insights,
            riskFactors={
                "finalRiskScore": fraud_decision.get("finalRiskScore", 0.0),
                "mlFraudScore": prediction.mlFraudScore,
                "ragCasesFound": prediction.ragContext.get("casesFound", 0)
                if prediction.ragContext else 0,
            },
            analyzedAt=datetime.now().replace(tzinfo=None).isoformat(),
        )

        self.producer.produce(
            topic=settings.topic_fraud_decisions,
            key=output.transactionId,
            value=output.model_dump_json(),
            callback=self._delivery_callback,
        )
        self.producer.poll(0)
        logger.debug(
            "fraud_decision_delivered",
            transaction_id=output.transactionId,
            is_fraudulent=output.isFraudulent,
            confidence=round(output.confidenceScore, 3),
        )

    def _risk_to_recommendation(self, risk_score: float) -> str:
        if risk_score >= 0.8:
            return "FRAUD_ALERT"
        if risk_score >= 0.6:
            return "HUMAN_REVIEW"
        return "APPROVE"

    def _persist_to_postgres(
        self,
        enriched: EnrichedTransaction,
        prediction: MLPrediction,
        fraud_decision: dict,
        shap_top3: dict,
        latency_ms: int
    ):
        """
        Dual-write pattern: persist to PostgreSQL for long-term audit trail.

        Saves:
        - Transaction record (immutable audit trail)
        - ML prediction (XGBoost score + SHAP values)
        - Fraud decision (LLM consensus + RAG context)
        """
        if not self.postgres_client:
            return  # Postgres not available, skip persistence

        try:
            # 1. Save transaction
            self.postgres_client.save_transaction({
                'transaction_id': enriched.transaction.transactionId,
                'customer_id': enriched.transaction.customerId,
                'amount': enriched.transaction.amount,
                'merchant_id': enriched.transaction.merchantId,
                'merchant_category': enriched.transaction.merchantCategory,
                'location': enriched.transaction.location,
                'device_id': enriched.transaction.metadata.get('deviceId'),
                'timestamp': enriched.transaction.timestamp.replace(tzinfo=None),
                'velocity_5min': enriched.velocityCount if enriched.velocityCount else 0,
                'avg_transaction_amount': (
                    enriched.customerProfile.averageTransactionAmount
                    if enriched.customerProfile else None
                ),
                'account_age_days': None,  # Not available in CustomerProfile schema
            })

            # 2. Save ML prediction
            shap_list = [{'feature': k, 'value': v} for k, v in shap_top3.items()]
            self.postgres_client.save_model_prediction({
                'transaction_id': enriched.transaction.transactionId,
                'model_type': 'xgboost',
                'prediction_score': prediction.mlFraudScore,
                'top_features': shap_list,
                'prediction_time_ms': latency_ms,
                'model_version': prediction.modelVersion,
            })

            # 3. Save fraud decision
            self.postgres_client.save_fraud_decision({
                'transaction_id': enriched.transaction.transactionId,
                'ml_score': prediction.mlFraudScore,
                'ml_model_type': 'xgboost',
                'shap_top_features': shap_list,
                'llm_confidence': fraud_decision.get('confidenceScore', 0.0),
                'fraud_pattern': fraud_decision.get('fraudPattern', 'unknown'),
                'reasoning': fraud_decision.get('explanation', ''),
                'recommendation': self._risk_to_recommendation(
                    fraud_decision.get('confidenceScore', 0.0) / 100
                ),
                'rag_context_used': prediction.ragContext.get('casesFound', 0) > 0 if prediction.ragContext else False,
                'rag_similarity_score': (
                    prediction.ragContext.get('maxSimilarity', 0.0)
                    if prediction.ragContext and prediction.ragContext.get('casesFound', 0) > 0
                    else None
                ),
                'rag_cases_retrieved': prediction.ragContext.get('casesFound', 0) if prediction.ragContext else 0,
                'decision': self._risk_to_recommendation(
                    fraud_decision.get('confidenceScore', 0.0) / 100
                ).replace('_ALERT', '').replace('_', ''),  # Convert to BLOCK/REVIEW/APPROVE
            })

            logger.debug(
                "postgres_persistence_complete",
                transaction_id=enriched.transaction.transactionId
            )

        except Exception as e:
            logger.error(
                "postgres_persistence_failed",
                transaction_id=enriched.transaction.transactionId,
                error=str(e),
                hint="Kafka streaming continues unaffected"
            )

if __name__ == "__main__":
    service = MLInferenceService()
    service.run()
