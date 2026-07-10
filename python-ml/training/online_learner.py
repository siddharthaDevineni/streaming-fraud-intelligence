"""
Online Learning Service
=======================
Consumes: analyst-feedback (auto-published by FraudStreams.java)
Updates:  River incremental model — no full retraining needed

This is what makes the system self-improving.
Every fraud decision automatically feeds back into the model.
When a human analyst later confirms or corrects a decision,
that label updates the model in real-time.
"""

import json

import joblib
import structlog
from config import settings
from confluent_kafka import Consumer, KafkaError
from river import ensemble, metrics

logger = structlog.get_logger()


class OnlineLearningService:

    def __init__(self):
        self.consumer = self._create_consumer()
        self.xgb_model = self._load_xgb_model()
        self.river_model = self._init_river_model()
        self.scaler = self._load_scaler()

        # Track online learning metrics
        self.metric_auc = metrics.ROCAUC()
        self.updates_count = 0

        logger.info("online_learning_service_ready",
                    river_model=type(self.river_model).__name__)

    def _create_consumer(self) -> Consumer:
        return Consumer({
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": settings.consumer_group_training,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        })

    def _load_xgb_model(self):
        try:
            model = joblib.load(settings.xgboost_model_path)
            logger.info("xgb_model_loaded_for_reference")
            return model
        except FileNotFoundError:
            logger.warning("xgb_model_not_found")
            return None

    def _load_scaler(self):
        try:
            return joblib.load(settings.scaler_path)
        except FileNotFoundError:
            logger.warning("scaler_not_found")
            return None

    def _init_river_model(self):
        """
        River's AdaptiveRandomForest — designed specifically for
        concept drift in streaming data. Unlike XGBoost, which is
        static after training, ARF continuously adapts.

        They complement each other.
        """
        return ensemble.SRPClassifier(
            n_models=10,
            seed=42
        )

    def _extract_features_from_feedback(self, feedback: dict) -> dict | None:
        """
        Feedback from Java contains transactionId and decision metadata.
        We use the confidence and agent consensus as proxy features
        for the online model since we don't have raw features here.
        """
        try:
            return {
                "confidence": feedback.get("confidence", 0.5),
                "agent_consensus": feedback.get("agentConsensus", 0),
                "ml_fraud_score": feedback.get("mlFraudScore", 0.5),
                "source_auto": 1 if feedback.get("source") == "AUTO_SYSTEM" else 0,
            }
        except Exception as e:
            logger.error("feature_extraction_failed", error=str(e))
            return None

    def _update_river_model(self, features: dict, label: bool):
        """
        River's learn_one() updates the model with a single example.
        No batch needed, no redeployment needed.
        This is the core of online learning.
        """
        self.river_model.learn_one(features, int(label))
        self.updates_count += 1

        # Update rolling AUC metric
        prediction = self.river_model.predict_proba_one(features)
        fraud_prob = prediction.get(1, 0.5)
        self.metric_auc.update(int(label), fraud_prob)

        if self.updates_count % 100 == 0:
            logger.info(
                "online_learning_progress",
                updates=self.updates_count,
                rolling_auc=round(self.metric_auc.get(), 4),
            )

    def run(self):
        self.consumer.subscribe([settings.topic_analyst_feedback])
        logger.info("online_learner_started",
                    topic=settings.topic_analyst_feedback)

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

                self._process_feedback(msg)

        except KeyboardInterrupt:
            logger.info("shutting_down",
                        total_updates=self.updates_count)
        finally:
            self.consumer.close()

    def _process_feedback(self, msg):
        try:
            feedback = json.loads(msg.value().decode("utf-8"))
        except json.JSONDecodeError as e:
            logger.error("deserialization_failed", error=str(e))
            return

        transaction_id = feedback.get("transactionId", "unknown")

        # Use actualFraud if human analyst provided it,
        # fall back to predictedFraud from the system
        actual_fraud = feedback.get("actualFraud")
        predicted_fraud = feedback.get("predictedFraud", False)

        if actual_fraud is not None:
            # Human analyst confirmed or corrected the decision
            label = actual_fraud
            label_source = "HUMAN_CONFIRMED"
        else:
            # Auto feedback — system's own prediction
            label = predicted_fraud
            label_source = "AUTO_PREDICTED"

        features = self._extract_features_from_feedback(feedback)
        if features is None:
            return

        self._update_river_model(features, label)

        logger.info(
            "online_model_updated",
            transaction_id=transaction_id,
            label=label,
            label_source=label_source,
            updates_total=self.updates_count,
        )


if __name__ == "__main__":
    service = OnlineLearningService()
    service.run()