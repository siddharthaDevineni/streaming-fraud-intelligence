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

import joblib
import json
import numpy as np
import shap
import structlog
import time
from agents.rag_retriever import RAGRetriever
from config import settings
from confluent_kafka import Consumer, Producer, KafkaError
from datetime import datetime, timezone
from features.engineer import extract_features, FEATURE_COLUMNS
from models.schemas import EnrichedTransaction, MLPrediction
from pydantic import ValidationError

logger = structlog.get_logger()


class MLInferenceService:

    def __init__(self):
        self.rag_retriever = RAGRetriever()
        self.consumer = self._create_consumer()

        self.producer = self._create_producer()
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        self.explainer = shap.TreeExplainer(self.model)
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

    def _process_message(self, msg):
        start_ms = time.time() * 1000

        try:
            # Deserialize — Pydantic validates the schema from Java
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
            # Feature engineering
            features = extract_features(enriched)

            # ML inference + SHAP explanation
            fraud_score, shap_top3 = self._score(features)

            # RAG retrieval — find similar confirmed fraud cases
            similar_cases = self.rag_retriever.retrieve(
                features=features,
                customer_id=enriched.transaction.customerId,
                n_results=3,
            )
            rag_context = self.rag_retriever.format_for_ml_prediction(similar_cases)

            latency_ms = int(time.time() * 1000 - start_ms)

            prediction = MLPrediction(
                transactionId=enriched.transaction.transactionId,
                customerId=enriched.transaction.customerId,
                mlFraudScore=fraud_score,
                lstmSequenceScore=0.0,  # LSTM added in next iteration
                combinedScore=fraud_score,  # will be weighted blend once LSTM is added
                modelVersion="xgb-v1",
                shapExplanation=shap_top3,
                featuresUsed=features,
                ragContext=rag_context,
                inferenceLatencyMs=latency_ms,
                timestamp=datetime.now(timezone.utc),
            )

            self._publish_prediction(prediction)

            logger.info(
                "inference_complete",
                transaction_id=enriched.transaction.transactionId,
                fraud_score=round(fraud_score, 3),
                latency_ms=latency_ms,
                top_shap_feature=list(shap_top3.keys())[0] if shap_top3 else None,
            )

        except Exception as e:
            logger.error("inference_failed",
                         transaction_id=enriched.transaction.transactionId,
                         error=str(e))

if __name__ == "__main__":
    service = MLInferenceService()
    service.run()
