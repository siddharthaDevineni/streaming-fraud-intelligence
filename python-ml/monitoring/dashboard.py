"""
Streaming Fraud Intelligence Dashboard
=======================================
Real-time visualization of the full fraud detection pipeline.

Shows:
  - Live transaction feed with ML scores, RAG context, agent consensus
  - Decision distribution (Fraud / Review / Approved)
  - RAG pattern distribution from ChromaDB
  - Pipeline architecture sidebar with health indicators

Run:
    cd ~/projects/streaming-fraud-intelligence/python-ml
    source .venv/bin/activate
    streamlit run monitoring/dashboard.py

Consumes:
  - ml-predictions   (XGBoost scores + RAG context from Python)
  - analyst-feedback (final agent decisions from Java)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
from collections import deque
from datetime import datetime, timezone

import chromadb
import pandas as pd
import plotly.express as px
import streamlit as st
from confluent_kafka import Consumer, KafkaError

from config import settings

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fraud Intelligence Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Singleton resources ───────────────────────────────────────────────────────
# @st.cache_resource creates ONE instance per session that survives reruns.
# This is critical for Kafka consumers — you must not create a new consumer
# on every rerun or you lose offset continuity.

@st.cache_resource
def get_consumer() -> Consumer:
    consumer = Consumer({
        "bootstrap.servers": settings.kafka_bootstrap_servers,
        "group.id": "streamlit-dashboard",
        "auto.offset.reset": "latest",   # only new messages — don't replay history
        "enable.auto.commit": True,
    })
    consumer.subscribe([
        settings.topic_ml_predictions,
        settings.topic_analyst_feedback,
    ])
    return consumer


@st.cache_resource
def get_chromadb_collection():
    """Connect to ChromaDB — returns None if collection doesn't exist yet."""
    try:
        client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        return client.get_collection(name=settings.chroma_collection_fraud)
    except Exception:
        return None


# ── Session state initialization ──────────────────────────────────────────────
# st.session_state persists across reruns within the same browser session.
# We use it to accumulate transactions over time.

def init_session_state():
    defaults = {
        "transactions":      deque(maxlen=100),  # rolling window of last 100
        "ml_cache":          {},                 # transactionId → ml prediction
        "fraud_count":       0,
        "review_count":      0,
        "approved_count":    0,
        "rag_hit_count":     0,
        "total_processed":   0,
        "refresh_rate":      2,
        "pending_feedback":  {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ── Kafka polling ─────────────────────────────────────────────────────────────

def poll_kafka(consumer: Consumer, max_messages: int = 30):
    """
    Poll Kafka for new messages and update session state.
    Called on every rerun — accumulates data over time.
    """
    for _ in range(max_messages):
        msg = consumer.poll(timeout=0.05)
        if msg is None:
            break
        if msg.error():
            if msg.error().code() != KafkaError._PARTITION_EOF:
                st.toast(f"Kafka error: {msg.error()}", icon="⚠️")
            continue

        try:
            raw = json.loads(msg.value().decode("utf-8"))
            topic = msg.topic()

            if topic == settings.topic_ml_predictions:
                _cache_ml_prediction(raw)
            elif topic == settings.topic_analyst_feedback:
                _process_feedback(raw)

        except (json.JSONDecodeError, KeyError):
            continue


def _cache_ml_prediction(raw: dict):
    """Store ML prediction keyed by transactionId for later join with feedback."""
    txn_id = raw.get("transactionId")
    if not txn_id:
        return

    rag_cases = raw.get("ragContext", {}).get("casesFound", 0)

    st.session_state.ml_cache[txn_id] = raw

    # Check if analyst-feedback arrived before this ml-prediction
    if txn_id in st.session_state.pending_feedback:
        pending = st.session_state.pending_feedback.pop(txn_id)
        _process_feedback(pending)

    # Keep cache bounded — remove oldest entries beyond 500
    if len(st.session_state.ml_cache) > 500:
        oldest_keys = list(st.session_state.ml_cache.keys())[:-500]
        for k in oldest_keys:
            del st.session_state.ml_cache[k]


def _process_feedback(raw: dict):
    """
    Join analyst-feedback with cached ml-prediction to build a display record.
    This is the same join pattern as feedback_embedder.py — feedback arrives
    before ml-predictions in the same poll batch, so we cache and retry.
    """
    txn_id = raw.get("transactionId")
    if not txn_id:
        return

    ml = st.session_state.ml_cache.get(txn_id, {})

    if not ml:
        # ml-predictions not cached yet — store for when it arrives
        st.session_state.pending_feedback[txn_id] = raw
        return
    rag_context = ml.get("ragContext", {})
    cases_found = rag_context.get("casesFound", 0)
    cases = rag_context.get("cases", [])

    predicted_fraud = raw.get("predictedFraud", False)
    confidence = raw.get("confidence", 0.0)
    agent_consensus = raw.get("agentConsensus", 0)

    # Determine decision bucket — matches Java branch logic
    if predicted_fraud and confidence >= 0.90:
        decision = "🚨 FRAUD"
        st.session_state.fraud_count += 1
    elif predicted_fraud:
        decision = "⚠️ REVIEW"
        st.session_state.review_count += 1
    else:
        decision = "✅ APPROVED"
        st.session_state.approved_count += 1

    if cases_found > 0:
        st.session_state.rag_hit_count += 1

    st.session_state.total_processed += 1

    top_case = cases[0] if cases else {}

    record = {
        "time":            datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "transactionId":   txn_id,                           # full id for detail lookup
        "txnShort":        txn_id[-8:],                      # last 8 chars for table display
        "customerId":      ml.get("customerId", "—"),
        "mlScore%":        round(ml.get("mlFraudScore", 0) * 100, 1),
        "ragCases":        cases_found,
        "topSimilarity%":  round(top_case.get("similarity", 0) * 100, 1)
        if cases_found > 0 else 0,
        "topPattern":      top_case.get("fraudPattern", "—")
        if cases_found > 0 else "—",
        "agents":          agent_consensus,
        "confidence%":     round(confidence * 100, 1),
        "decision":        decision,
    }

    st.session_state.transactions.appendleft(record)


# ── UI components ─────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.header("Architecture")

        st.markdown("**Running Services:**")
        st.markdown("🟢 Kafka Streams `Java`")
        st.markdown("🟢 XGBoost Inference `Python`")
        st.markdown("🟢 RAG Retriever (ChromaDB) `Python`")
        st.markdown("🟢 Feedback Embedder `Python`")
        st.markdown("🟢 River Online Learner `Python`")

        st.divider()

        collection = get_chromadb_collection()
        if collection:
            count = collection.count()
            st.metric("ChromaDB Cases", count)
        else:
            st.warning("ChromaDB empty — run feedback_embedder.py first")

        st.divider()

        st.markdown("**Data Flow**")

        stages = [
            ("📥", "transactions", "raw Kafka topic"),
            ("⚙️", "Kafka Streams", "velocity + profile enrichment"),
            ("🔀", "enriched-transactions", "Java → Python bridge"),
            ("🧠", "XGBoost + RAG", "ML score + similar cases"),
            ("📊", "ml-predictions", "Python → Java bridge"),
            ("🤖", "LLM Agents", "5 agents reason in parallel"),
            ("📤", "analyst-feedback", "final fraud decision"),
            ("📚", "ChromaDB embed", "confirmed cases → memory"),
        ]

        for i, (icon, name, desc) in enumerate(stages):
            st.markdown(
                f"{icon} **{name}**  \n"
                f"<span style='color:gray;font-size:0.8em'>{desc}</span>",
                unsafe_allow_html=True,
            )
            if i < len(stages) - 1:
                st.markdown(
                    "<div style='text-align:center;color:gray'>↓</div>",
                    unsafe_allow_html=True,
                )

        st.divider()

        st.session_state.refresh_rate = st.slider(
            "Refresh (seconds)", min_value=1, max_value=10, value=2
        )

        if st.button("🗑️ Clear display"):
            st.session_state.transactions.clear()
            st.session_state.ml_cache.clear()
            st.session_state.fraud_count = 0
            st.session_state.review_count = 0
            st.session_state.approved_count = 0
            st.session_state.rag_hit_count = 0
            st.session_state.total_processed = 0
            st.rerun()


def render_header():
    st.title("🛡️ Streaming Fraud Intelligence Dashboard")
    st.caption(
        "Kafka Streams · XGBoost · ChromaDB RAG · River Online Learning  |  "
        f"Auto-refreshing every {st.session_state.refresh_rate}s"
    )


def render_metrics():
    total = st.session_state.total_processed or 1
    rag_rate = round(st.session_state.rag_hit_count / total * 100, 1)
    fraud_rate = round(st.session_state.fraud_count / total * 100, 1)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Processed", st.session_state.total_processed)
    col2.metric("🚨 Fraud Alerts", st.session_state.fraud_count,
                delta=f"{fraud_rate}% rate")
    col3.metric("⚠️ Human Review", st.session_state.review_count)
    col4.metric("✅ Approved", st.session_state.approved_count)
    col5.metric("🧠 RAG Hit Rate", f"{rag_rate}%",
                delta="cases retrieved" if rag_rate > 0 else "cold start")

    collection = get_chromadb_collection()
    col6.metric("📚 ChromaDB", collection.count() if collection else 0,
                delta="confirmed cases")


def render_transaction_feed():
    st.subheader("Live Transaction Feed")

    if not st.session_state.transactions:
        st.info(
            "Waiting for transactions… "
            "Start all services and run `TestDataGenerator` scenario 2."
        )
        return

    # Build display dataframe — drop full transactionId, show short version
    display_cols = [
        "time", "txnShort", "customerId",
        "mlScore%", "ragCases", "topSimilarity%",
        "topPattern", "agents", "confidence%", "decision"
    ]
    df = pd.DataFrame(list(st.session_state.transactions))[display_cols]
    df.columns = [
        "Time", "TXN", "Customer",
        "ML Score%", "RAG Cases", "Similarity%",
        "Pattern", "Agents", "Confidence%", "Decision"
    ]

    def color_decision(val):
        if "FRAUD" in str(val):
            return "color: #ff4444; font-weight: bold"
        if "REVIEW" in str(val):
            return "color: #ffaa00; font-weight: bold"
        return "color: #44cc44; font-weight: bold"

    styled = df.style.map(color_decision, subset=["Decision"])
    st.dataframe(styled, use_container_width=True, height=380)


def render_charts():
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Decision Distribution")
        fraud = st.session_state.fraud_count
        review = st.session_state.review_count
        approved = st.session_state.approved_count

        if fraud + review + approved > 0:
            fig = px.pie(
                values=[fraud, review, approved],
                names=["Fraud", "Review", "Approved"],
                color=["Fraud", "Review", "Approved"],
                color_discrete_map={
                    "Fraud":    "#ff4444",
                    "Review":   "#ffaa00",
                    "Approved": "#44bb44",
                },
                hole=0.4,
            )
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=260)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No decisions yet")

    with col2:
        st.subheader("RAG Pattern Distribution")
        collection = get_chromadb_collection()

        if collection and collection.count() > 0:
            results = collection.get(include=["metadatas"])
            patterns: dict = {}
            for meta in results["metadatas"]:
                p = meta.get("fraudPattern", "unknown")
                patterns[p] = patterns.get(p, 0) + 1

            fig = px.bar(
                x=list(patterns.keys()),
                y=list(patterns.values()),
                labels={"x": "Fraud Pattern", "y": "Confirmed Cases"},
                color=list(patterns.values()),
                color_continuous_scale="Reds",
                text_auto=True,
            )
            fig.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=260,
                showlegend=False,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No ChromaDB cases yet — process some transactions first")


def render_rag_impact():
    """
    Show the measurable impact of RAG on agent scores.
    Compares transactions WITH and WITHOUT RAG context.
    """
    st.subheader("RAG Impact on Detection")

    txns = list(st.session_state.transactions)
    if len(txns) < 2:
        st.info("Need at least 2 transactions to compare RAG impact")
        return

    with_rag    = [t for t in txns if t["ragCases"] > 0]
    without_rag = [t for t in txns if t["ragCases"] == 0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Without RAG**")
        if without_rag:
            avg_conf = sum(t["confidence%"] for t in without_rag) / len(without_rag)
            st.metric("Avg Confidence", f"{round(avg_conf, 1)}%")
            st.metric("Transactions", len(without_rag))
        else:
            st.info("None yet")

    with col2:
        st.markdown("**With RAG**")
        if with_rag:
            avg_conf = sum(t["confidence%"] for t in with_rag) / len(with_rag)
            avg_sim  = sum(t["topSimilarity%"] for t in with_rag) / len(with_rag)
            st.metric("Avg Confidence", f"{round(avg_conf, 1)}%")
            st.metric("Avg Similarity", f"{round(avg_sim, 1)}%")
            st.metric("Transactions", len(with_rag))
        else:
            st.info("None yet — ChromaDB needs confirmed cases first")

    with col3:
        st.markdown("**Improvement**")
        if with_rag and without_rag:
            avg_with    = sum(t["confidence%"] for t in with_rag) / len(with_rag)
            avg_without = sum(t["confidence%"] for t in without_rag) / len(without_rag)
            delta = round(avg_with - avg_without, 1)
            st.metric(
                "Confidence Delta",
                f"+{delta}%" if delta > 0 else f"{delta}%",
                delta=f"{'↑ RAG helps' if delta > 0 else '→ No difference'}",
            )
        else:
            st.info("Need both states to compare")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    init_session_state()
    render_sidebar()

    consumer = get_consumer()
    poll_kafka(consumer)

    render_header()
    render_metrics()

    st.divider()
    render_transaction_feed()

    st.divider()
    col1, col2 = st.columns([1, 1])
    with col1:
        render_charts()
    with col2:
        render_rag_impact()

    # Auto-refresh — sleep then rerun the entire script
    time.sleep(st.session_state.refresh_rate)
    st.rerun()


if __name__ == "__main__":
    main()