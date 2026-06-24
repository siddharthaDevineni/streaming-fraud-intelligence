from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Kafka bootstrap
    kafka_bootstrap_servers: str = "localhost:9092"

    # Topics
    topic_enriched_transactions: str = "enriched-transactions"
    topic_ml_predictions: str = "ml-predictions"
    topic_analyst_feedback: str = "analyst-feedback"
    topic_model_health: str = "model-health"

    # Consumer groups
    consumer_group_inference: str = "ml-inference-service"
    consumer_group_training: str = "ml-training-service"

    # Groq / LLM
    groq_api_key: str = "dummy-key"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.1-8b-instant"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_fraud: str = "confirmed_fraud_cases"

    # Model paths
    xgboost_model_path: str = "./models/fraud_xgb.pkl"
    lstm_model_path: str = "./models/fraud_lstm.keras"
    scaler_path: str = "./models/scaler.pkl"

    # Thresholds
    high_velocity_threshold: int = 3
    fraud_risk_threshold: float = 0.6
    high_confidence_threshold: float = 0.8

    class Config:
        env_file = "../.env"


settings = Settings()