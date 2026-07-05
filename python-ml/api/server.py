from datetime import datetime
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from langchain_groq import ChatGroq
from pydantic import BaseModel

from agents.agent_coordinator import AgentCoordinator
from agents.rag_retriever import RAGRetriever
from config import settings
from features.engineer import extract_features
from models.schemas import (
    CustomerProfile,
    EnrichedTransaction,
    Transaction,
)
from utils.pipeline_utils import build_streaming_context, enriched_to_dict

logger = structlog.get_logger()

# ── Shared resources (initialised once at startup) ────────────────────────────

agent_coordinator = AgentCoordinator()
rag_retriever = RAGRetriever()
chat_llm = ChatGroq


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy resources once at startup, release at shutdown."""
    global agent_coordinator, rag_retriever, chat_llm

    logger.info("fastapi_startup")
    agent_coordinator = AgentCoordinator()
    rag_retriever = RAGRetriever()
    chat_llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.1,   # lower = more factual, less creative
        max_tokens=1024,   # enough for a detailed answer, not runaway
    )
    logger.info("fastapi_ready")

    yield

    logger.info("fastapi_shutdown")


app = FastAPI(title="Streaming Fraud Intelligence - Sync API",
              description=("Synchronous REST API for ad-hoc fraud analysis. "
                           "The Kafka Streams pipeline handles real-time traffic; "
                           "this API serves manual transaction checks and to interact with the system."),
              version="2.0.0",
              lifespan=lifespan)

# ── Request / Response models ─────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    transaction: Transaction
    customerProfile: CustomerProfile | None = None
    velocityCount: int | None = None

class ChatRequest(BaseModel):
    question: str

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/analyze", summary="Full ML ++ agent pipeline for a single transaction")
def analyze_transaction(request: AnalyzeRequest):
    """Synchronous fraud analysis — mirrors the Kafka Streams pipeline
    but for ad-hoc or interactive use.

    Called by FraudDetectionController.java via RestClient."""
    txn = request.transaction
    logger.info("analyze_requested", transaction_id=txn.transactionId)

    try:
        # Build a minimal EnrichedTransaction for feature engineering
        enriched = EnrichedTransaction(
            transaction=txn,
            customerProfile=request.customerProfile,
            velocityCount=request.velocityCount
        )
        features = extract_features(enriched)

        # RAG retrieval
        similar_cases = rag_retriever.retrieve(
            features=features,
            customer_id=txn.customerId,
            n_results=3,
        )
        rag_context_text = rag_retriever.format_for_agent_prompt(similar_cases)

        # Build streaming context string for agents
        streaming_context = build_streaming_context(
            enriched=enriched,
            rag_context_text=rag_context_text,
        )

        transaction_dict = enriched_to_dict(enriched, features)
        decision = agent_coordinator.investigate(
            transaction=transaction_dict,
            streaming_context=streaming_context,
            has_high_velocity=(request.velocityCount or 0) >= settings.high_velocity_threshold,
            has_customer_profile=request.customerProfile is not None,
        )

        logger.info(
            "analyze_complete",
            transaction_id=txn.transactionId,
            is_fraudulent=decision["isFraudulent"],
            confidence=decision["confidenceScore"],
            agents=decision["agentCount"],
        )

        return decision

    except Exception as e:
        logger.error(
            "analyze_failed",
            transaction_id=txn.transactionId,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/investigation/chat", summary="Conversational Q&A about the fraud detection system")
def chat_with_system(request: ChatRequest):
    """
    General conversational endpoint — answers questions about how the
    multi-agent system works, fraud patterns, or detection methodology.

    Not transaction-specific. Called by ConversationalController.java.
    """
    logger.info("chat_requested", question=request.question[:100])

    system_prompt = f"""You are an expert fraud investigation assistant. 
    Answer questions about our fraud detection system accurately based ONLY 
    on the architecture described below. Do not invent or assume anything 
    not described here.
    
    EXACT SYSTEM ARCHITECTURE:
    
    Java Kafka Streams (streaming infrastructure):
    - Sub-topology 0: selectKey(customerId) → repartition
    - Sub-topology 1: windowedBy(5min).count() → RocksDB velocity store
    - Sub-topology 2: leftJoin(customerProfiles) + leftJoin(velocityContext)
      → enriched-transactions topic (Java → Python bridge)
    - Reads fraud-decisions topic (Python → Java) → routes to:
      fraud-alerts, human-review, approved-transactions, analyst-feedback
    
    Python ML layer (intelligence):
    - XGBoost inference: 19 Polars features, StandardScaler, SHAP TreeExplainer
    - ChromaDB RAG: sentence-transformers all-MiniLM-L6-v2 (384-dim, local),
      retrieves top-3 similar confirmed fraud cases before agent analysis
    - LangChain agents via Groq API
    - River SRPClassifier: online learning from analyst-feedback topic
    
    EXACT AGENT PIPELINE (10 LLM calls per transaction, 3 phases):
    
    Phase 1 — 5 agents run IN PARALLEL simultaneously (not sequentially):
      - BehaviorAnalyst (weight 1.2x): customer velocity + spending deviation
      - PatternDetector (weight 1.3x): known attack signatures (card testing, vpn_bot_fraud, account_takeover)
      - RiskAssessor (weight 1.1x): financial risk + customer profile baseline
      - GeographicAnalyst (weight 1.0x): location anomaly + VPN/proxy detection
      - TemporalAnalyst (weight 1.0x): timing patterns + bot behavior indicators
      All 5 run simultaneously via ThreadPoolExecutor — not one after another.
    
    Phase 2 — collaboration runs IN PARALLEL if conditions are met:
      IF high velocity detected (velocityCount >= 3):
        PatternDetector + TemporalAnalyst collaborate (+2 LLM calls)
      IF customer profile exists:
        BehaviorAnalyst + RiskAssessor collaborate (+2 LLM calls)
      Both collaboration rounds run simultaneously if both conditions are true.
    
    Phase 2c — consensus (always, 1 LLM call):
      STREAMING_CONSENSUS_ORCHESTRATOR reads all Phase 1 + Phase 2 insights
      and produces a final weighted risk score (weight 0.8x).
    
    Phase 3 — decision synthesis (no LLM call):
      calculateWeightedRiskScore() across all insights
      + streaming intelligence bonus:
        high velocity: +0.25, unusual amount: +0.20, HIGH risk customer: +0.10
      → isFraudulent, confidenceScore, FraudDecision published to Kafka
    
    ChromaDB RAG context is injected into ALL agent prompts in Phase 1 and 2
    BEFORE the agents reason — not after. This gives every agent historical
    precedent from confirmed fraud cases at the moment of analysis.
    
    Question: {request.question}
    
    Answer accurately based only on the architecture above. If the question 
    asks about something not described here, say so rather than inventing details.
    
    Answer in clear prose paragraphs. Do not use markdown headers or bullet 
    points — write as if explaining to a technical colleague verbally"""

    try:
        response = chat_llm.invoke(system_prompt,
                                   stop=None # no early stopping
                                   )

        logger.info("chat_complete")

        return {
            "response": response.content,
            "timestamp": datetime.now().isoformat(),
            "systemCapabilities" : [
                "Real-time Kafka Streams enrichment (Java)",
                "XGBoost + SHAP ML inference (Python)",
                "ChromaDB RAG historical case retrieval (Python)",
                "10 LLM calls per transaction — 5 parallel + 4 collaborative + 1 consensus",
                "River SRPClassifier online learning (Python)",
                "LangSmith full pipeline observability"
            ]
        }

    except Exception as e:
        logger.error("chat_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", summary="Health check")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "agent_coordinator": "ready",
            "rag_retriever": "ready",
            "chat_llm": "ready",
        },
    }

if __name__ == "__main__":
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

