# Streaming Fraud Intelligence

> **Kafka Streams + XGBoost + LLM Agents + RAG — a self-improving fraud detection pipeline**

A hybrid Java/Python fraud detection system combining Kafka Streams real-time enrichment, XGBoost ML inference with SHAP explainability, ChromaDB RAG (historical fraud case retrieval), multi-agent LLM analysis, and River online learning — all wired as a continuous feedback loop.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Siddhartha_Devineni-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/siddhartha-devineni/)
[![Medium](https://img.shields.io/badge/Medium-Article-black?style=flat&logo=medium)](https://medium.com/@siddhartha.devineni/kafka-streams-make-ai-agents-fraud-detection-smarter-55fce4d6be3a)
[![Dev.to](https://img.shields.io/badge/dev.to-Article-0A0A0A?style=flat&logo=devdotto)](https://dev.to/siddhartha_devineni_896e9/kafka-streams-make-ai-agents-fraud-detection-smarter-24c1)

---

## See It In Action

<!-- REPLACE: record a 45-60s screen capture showing the Streamlit dashboard
     updating live as scenario 2 runs — transaction feed populating, RAG cases
     appearing, pattern chart building, +9.1% confidence delta showing.
     Convert to GIF: ffmpeg -i demo.mp4 -vf "fps=10,scale=1200:-1" -loop 0 docs/demo.gif
     Keep under 10MB or GitHub will not render it inline. -->

![Streaming Fraud Intelligence Dashboard](docs/gif_streaming_fraud_intelligence.gif)
> Real-time Streamlit dashboard: live transaction feed, ML scores, RAG similarity, fraud pattern distribution, and a before/after confidence comparison panel

![Dashboard screenshot](docs/Demo_scenario_2_1.png)

---

## What this project covers

Five layers running concurrently and feeding each other:

1. **Kafka Streams** `Java` — real-time velocity windows, customer profile
   KTable joins, enriched-transactions topic
2. **XGBoost + SHAP** `Python` — ML inference with explainability, 19
   engineered features via Polars
3. **ChromaDB RAG** `Python` — confirmed fraud cases embedded via
   sentence-transformers, retrieved at inference time to give agents
   historical context from your own system's confirmed history
4. **LLM Agents** `Java / Groq` — 5 specialized agents running 10 LLM
   calls per transaction across parallel and collaborative phases
5. **River Online Learning** `Python` — SRPClassifier updating from the
   analyst-feedback Kafka topic in real time, no retraining cycle required

Each layer feeds the next and the last feeds back into the first —
analyst decisions flow back through Kafka into ChromaDB and River,
making the system more accurate over time without manual intervention.

---

## The Problem

**Rule-based systems** catch known patterns but miss novel attacks:
- Rules are static — new fraud vectors require manual updates that lag weeks behind attackers
- Context is ignored — a `$30` transaction triggers or does not trigger the same rule
  regardless of whether the customer's average is `$20` or `$2,000`
- Combinatorial explosion — detecting fraud that requires 5! signals together means
  writing 5! rule combinations manually

**LLM-only systems without streaming context** reason well but reason blind:
- Every transaction is analyzed in isolation — no awareness of the last 5 minutes
- No customer baseline — an LLM cannot know whether $30 is normal or anomalous
  for a specific customer without real-time profile context
- No memory — without RAG, agents have no knowledge of confirmed fraud cases
  from your own system's history

**Example:** A $30 transaction looks normal in isolation. With streaming context:
- Customer average: $244 → amount is 87% below baseline
- 15 transactions in the last 5 minutes → velocity attack in progress
- Bot device ID + suspicious merchant + unknown location → automated attack

→ **Card testing attack detected at 95% confidence**

This system addresses both gaps: Kafka Streams provides the real-time context
that LLMs lack, and RAG provides the institutional memory that rules cannot encode.

---

## The Solution: Streaming-Intelligent AI

Kafka Streams enriches every transaction with real-time context **before** it reaches the LLM agents:

```java
// Traditional: transaction alone
Transaction = { amount: $30, merchant: "MERCHANT-SUSPICIOUS-7" }

// This system: streaming-enriched transaction
EnrichedTransaction = {
  transaction:     { amount: $30, merchant: "MERCHANT-SUSPICIOUS-7" },
  customerProfile: { average: $244, riskLevel: "HIGH" },
  velocityCount:   15,   // ← Kafka Streams 5-minute tumbling window
  mlFraudScore:    0.9998, // ← XGBoost inference from Python
  ragContext:      { casesFound: 3, topSimilarity: 82%, pattern: "card_testing" } // from Python
}
```
---

## Architecture

![diagram](docs/mermaid-diagram-2026-06-24-192254.png)

The diagram shows two concurrent paths from the `transactions` topic — the
**fan-out** that causes the stream-table join timing lag documented in the
trade-offs section. The left path (Steps 1–2) builds the velocity KTable via
Kafka Streams stateful processing. The right path (Step 1 fan-out) immediately
enriches transactions in Sub-topology 2, joining against the velocity KTable
via a RocksDB lookup — but Sub-topology 1 may not have committed the latest
count yet. The Python ML layer (Steps 6–8) runs as a separate process,
bridged via two Kafka topics: `enriched-transactions` (Java → Python) and
`ml-predictions` (Python → Java), closing a feedback loop that includes
ChromaDB RAG retrieval and River online learning.

---

## LLM Agent Pipeline (10 LLM calls per transaction)

Three phases in `AgentCoordinator.java`:

**Phase 1 — 5 specialized agents in parallel** (`CompletableFuture`, fixed thread pool of 5):

| Agent | Specialization | Weight |
|---|---|---|
| BehaviorAnalyst | Velocity + spending deviation | 1.2x |
| PatternDetector | Known attack signatures | 1.3x |
| RiskAssessor | Financial risk + customer profile | 1.1x |
| GeographicAnalyst | Location anomaly + VPN detection | 1.0x |
| TemporalAnalyst | Timing patterns + bot indicators | 1.0x |

**Phase 2 — 4 collaboration calls** (triggered when high velocity or customer profile is available — always true in production attack scenarios):
- PatternDetector + TemporalAnalyst debate the velocity question (+2 LLM calls)
- BehaviorAnalyst + RiskAssessor debate the customer profile question (+2 LLM calls)

**Phase 2c — 1 consensus call** (`STREAMING_CONSENSUS_ORCHESTRATOR`):
Reads all preceding insights and produces a final weighted risk score with weight `0.8` (lower than specialized agents, which range from 1.0–1.3x).

**Phase 3 — decision synthesis** (no additional LLM call):
- `calculateWeightedRiskScore()` across all 10 insights
- `calculateStreamingIntelligenceBonus()`: velocity +0.25, unusual amount +0.20, HIGH risk customer +0.10
- Routing: `finalRiskScore >= 0.6 && confidence > 0.8` → `fraud-alerts`

Total wall-clock time per transaction: **1.5–2.7 seconds**
(4 sequential phases, parallel within each phase, Groq API latency ~300–400ms per call)

---

## Python ML Layer

### Feature engineering (`features/engineer.py`)

19 features extracted from `EnrichedTransaction` using **Polars** (5–10× faster than Pandas for columnar operations):

```
amount_ratio, daily_limit_ratio, is_amount_unusual
velocity_count, velocity_squared, is_high_velocity
is_unknown_location, is_primary_location, is_different_city
is_online, is_typical_category, is_suspicious_merchant
hour, is_off_hours, is_weekend
is_high_risk_customer, is_low_risk_customer, is_bot_device, is_rapid_fire
```

### XGBoost inference + SHAP (`consumers/inference_consumer.py`)

- Pre-trained XGBoost model with StandardScaler applied at inference time (training-serving skew prevention)
- **SHAP TreeExplainer** produces top-3 feature contributions per prediction — used for both explainability and debugging
- Publishes `MLPrediction` to `ml-predictions` topic including `ragContext` field
- Trained with MLflow experiment tracking

### Online learning (`training/online_learner.py`)

- **River `SRPClassifier`** (Streaming Random Patches — designed specifically for concept drift in streaming data)
- `learn_one()` called per `analyst-feedback` message — model updates in microseconds, no retraining cycle
- Tracks rolling ROCAUC metric, logs progress every 100 updates

---

## RAG Pipeline (ChromaDB + sentence-transformers)


### Why RAG in a fraud detection system

Without RAG, agents reason from general LLM training knowledge —
they know what card testing "typically" looks like from their
training data, but they have no knowledge of the specific fraud
cases your system has already confirmed.

With RAG, agents reason from your system's own confirmed history.
When a new transaction arrives, the 3 most similar confirmed cases
are retrieved from ChromaDB and injected directly into every
agent's prompt:

```
SIMILAR CONFIRMED FRAUD CASES FROM HISTORY:

Case 1 (similarity: 82%, confirmed: 2026-06-13, by: system):
  fraud pattern: card_testing
  velocity: 15 transactions in 5 minutes (high velocity attack)
  merchant: suspicious online (atypical category)
  location: unknown/vpn
  device: bot device detected
  outcome: FRAUD CONFIRMED — 10 agents agreed, confidence 1.00

Case 2 (similarity: 79%, confirmed: 2026-06-14, by: system):
  fraud pattern: card_testing
  velocity: 15 transactions in 5 minutes (high velocity attack)
  merchant: suspicious online (atypical category)
  location: unknown/vpn
  device: bot device detected
  outcome: FRAUD CONFIRMED — 10 agents agreed, confidence 1.00

Case 3 (similarity: 76%, confirmed: 2026-06-15, by: system):
  fraud pattern: vpn_bot_fraud
  velocity: 1 transactions in 5 minutes
  merchant: suspicious online (atypical category)
  location: unknown/vpn
  device: bot device detected
  outcome: FRAUD CONFIRMED — 8 agents agreed, confidence 0.80

Use these cases as precedent. Higher similarity = more relevant to current transaction.
```
---

### How it works

**Embedding** (`agents/feedback_embedder.py`):
- Subscribes to both `ml-predictions` and `analyst-feedback` topics via a single consumer
- Embeds confirmed fraud cases using `sentence-transformers all-MiniLM-L6-v2` (local, free, 384-dimensional, <5ms per embedding on CPU)
- Stores in ChromaDB with metadata for filtered retrieval (by customer, by pattern)
- Pending-buffer pattern handles cross-topic ordering: `analyst-feedback` consistently arrives before its matching `ml-predictions` in the same poll cycle — feedback is buffered until features are available
- Self-bootstrapping: TXN #1 in an attack is embedded immediately, TXN #3–4 retrieves it in the same run

**Retrieval** (`agents/rag_retriever.py`):
- Called at inference time before publishing `MLPrediction`
- Dual-query: customer-specific cases (priority) + general similar cases (deduplicated)
- Minimum similarity threshold: 0.50 cosine distance
- Results travel in `ragContext` field through `ml-predictions` topic → Java `StreamingContext` → injected into all 5 agent prompts

### Measured impact

| Metric | Without RAG | With RAG |
|---|---|---|
| Avg agent confidence | 80.0% | 95.3% |
| Transactions | 1 | 17 |
| Avg top similarity | — | 73.3% |
| RAG hit rate | — | 94.4% |
| **Confidence delta** | — | **+15.3%** |

The "Without RAG" bucket consists of the first 1–3 transactions in a cold-start
run — processed before `ml-predictions` (carrying `ragContext`) becomes available
in the Java KTable join, consistent with the fan-out timing lag. From transaction
#3–4 onward, every transaction retrieves RAG context and benefits from confirmed
historical fraud cases.

Agent-level impact: `RISK_ASSESSOR` improved from **0.42 → 0.85** when confirmed
fraud cases were retrieved at 79–82% similarity. System-level: **+15.3% confidence
delta** across 17 transactions with RAG context vs 1 cold-start transaction without.

---

## Real-time Dashboard (Streamlit)

`python-ml/monitoring/dashboard.py` — auto-refreshes every 2s.

**Panels:**
- Live transaction feed with ML score, RAG cases, similarity%, fraud pattern, agent consensus, confidence, decision (colour-coded)
- Decision distribution donut chart (Fraud / Review / Approved)
- RAG pattern distribution bar chart from ChromaDB metadata (`card_testing`, `vpn_bot_fraud`, `account_takeover`)
- RAG Impact panel: Without RAG vs With RAG confidence comparison — live, measured from actual traffic in the current session

```bash
cd python-ml
streamlit run monitoring/dashboard.py
# opens at http://localhost:8501
```
---

## Real Detection Examples

### Example 1 — High-confidence fraud alert (100% confidence)

**AUTO-BLOCKED** → `fraud-alerts` topic

```
Customer: CUST-001 | avg $244 | HIGH risk
Transaction: $30 | MERCHANT-SUSPICIOUS-7 | ONLINE | Unknown Location
Context: 15 transactions in 5 minutes | BOT-DEVICE-1 | rapidFire=true
```

**Agent scores:**

| Agent | Risk | Key finding |
|---|---|---|
| BehaviorAnalyst | 95% | High velocity highly unusual, small amounts = detection avoidance |
| PatternDetector | 95% | Matches card testing: rapid + small + suspicious merchant |
| RiskAssessor | 85% | RAG: 3 confirmed card_testing cases at 82% similarity |
| GeographicAnalyst | 85% | Unknown location = VPN/proxy, geographic impossibility |
| TemporalAnalyst | 95% | Sub-second intervals = automated script |

**Result:** 10 agents agree (ratio: 1.0), base 0.88 + streaming bonus 0.35 = **100% confidence, auto-blocked**

---

### Example 2 — Medium-confidence human review (80% confidence)

**ANALYST REQUIRED** → `human-review` topic

```
Customer: CUST-001 | avg $392 | HIGH risk
Transaction: $30 | MERCHANT-SUSPICIOUS-4 | ONLINE | Unknown Location
Context: 1 transaction (no velocity alert yet)
```

**Why uncertain — agent disagreement:**

| Agent | Risk | Assessment |
|---|---|---|
| GeographicAnalyst | 85% | Location mismatch, VPN indicator |
| TemporalAnalyst | 85% | Suspicious timing (Sunday, unusual hours) |
| BehaviorAnalyst | 65% | Amount deviation + merchant mismatch |
| PatternDetector | 65% | Possible card testing, not confirmed |
| RiskAssessor | 32% | LOW financial impact, HIGH-risk customer expects anomalies |

Score range 32%–85% = high disagreement → system correctly recognises uncertainty → human review rather than auto-block.

---

### Example 3 — Auto-approved legitimate transaction (91% confidence)

**AUTO-APPROVED** → `approved-transactions` topic

```
Customer: CUST-001 | avg $178 | LOW risk
Transaction: $48 | Grocery Store | GROCERY | Houston (matches baseline)
Context: 1 transaction, normal velocity, typical category
```

All 5 agents score below fraud threshold. No velocity, no location mismatch, no suspicious merchant. Auto-approved without human review.

---

## Known Architectural Trade-offs

### Stream-table join timing lag

The velocity KTable and enrichment pipeline both consume from `transactions` simultaneously (fan-out topology). Sub-topology 2 reads the velocity KTable at the same moment Sub-topology 1 is still computing the current batch's velocity count — so the first 1–2 transactions in a burst see `velocity_count=1` rather than the accumulated count.

**Root cause:** `selectKey()` creates a repartition topic boundary (`KSTREAM-KEY-SELECT-0000000015-repartition`, visible in Kafka UI). Sub-topology 2 reads RocksDB before Sub-topology 1 has committed. With `COMMIT_INTERVAL_MS=1000ms` and transactions arriving every 200ms, state is stale for the first 1–2 transactions per commit cycle.

**Four-layer compensation mechanism:**

1. **KTable propagation** — `ml-predictions` KTable stores the latest ML score per customer. Once TXN #1 is correctly scored, subsequent transactions inherit that fraud signal
2. **Customer profile** — `customerProfiles` join is unaffected by the lag (direct KTable lookup, no repartition chain)
3. **Raw transaction features** — LLM agents reason from `merchantId`, `deviceId`, and `rapidFire` flag directly from the transaction record, independent of all state stores
4. **RAG historical context** — when XGBoost scores 0.0% due to stale velocity, agents correctly identify fraud using confirmed historical cases. Measured: 95.3% avg confidence across 17 transactions with `mlFraudScore=0.0%`
   and stale velocity, vs. 80.0% without RAG context — a **+15.3% delta**

**Production mitigation:** Reduce `COMMIT_INTERVAL_MS_CONFIG` from 1000ms to 100ms. Full elimination requires the Processor API with a Punctuator — intentionally deferred in favour of the four-layer compensation mechanism.

---

## Running Locally

### Prerequisites

- Java 21, Maven
- Python 3.12
- Docker + Docker Compose
- Groq API key (free tier at [console.groq.com](https://console.groq.com))

### 1 — Start Kafka

```bash
docker compose up -d
./setup-topics.sh
```

### 2 — Configure API key

```bash
export GROQ_API_KEY=your-groq-api-key
```

### 3 — Set up Python environment

```bash
cd python-ml
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
```

### 4 — Train XGBoost model (first time only)

```bash
python -m training.train_xgboost
# produces models/fraud_xgb.pkl and models/scaler.pkl
```

### 5 — Start all services (5 terminals)

```bash
# Terminal 1 — Kafka Streams + LLM Agents
mvn spring-boot:run

# Terminal 2 — XGBoost inference + RAG retrieval
cd python-ml && source .venv/bin/activate && python -m consumers.inference_consumer

# Terminal 3 — ChromaDB feedback embedder
python -m agents.feedback_embedder

# Terminal 4 — River online learner
python -m training.online_learner

# Terminal 5 — Streamlit dashboard
streamlit run monitoring/dashboard.py
```

### 6 — Generate test data

Run `TestDataGenerator.java` from your IDE.

**Recommended scenario: #2 — High velocity attack**
→ 15 rapid transactions, triggers all pipeline layers, best demonstrates RAG self-bootstrapping

### Full reset

```bash
docker compose down -v    # wipes Kafka volumes — eliminates producer epoch conflicts
docker compose up -d
./setup-topics.sh
rm -rf /tmp/kafka-streams/intelligent-fraud-detection/
rm -rf python-ml/chroma_db
# restart all 5 services
```

> **Important:** When wiping `chroma_db`, restart **all** processes that hold a ChromaDB connection (`inference_consumer.py`, `feedback_embedder.py`, Streamlit dashboard) — not just one. Linux keeps the old file inode alive for any process with an open file handle, so a partial restart means one process reads the new empty DB while others silently continue reading deleted data.

---

## Kafka Topics

**Input:**
- `transactions` — raw transaction events
- `customerProfiles` — customer baseline KTable

**Java → Python bridge:**
- `enriched-transactions` — transaction + customer profile + velocity context

**Python → Java bridge:**
- `ml-predictions` — XGBoost score + SHAP explanation + RAG context (consumed by Java via KTable join)

**Output:**
- `fraud-alerts` — high-confidence fraud, auto-block
- `human-review` — uncertain cases, analyst required
- `approved-transactions` — legitimate, auto-approved

**Feedback loop:**
- `analyst-feedback` — auto-sink from every `FraudDecision`, consumed by River online learner and ChromaDB embedder

**Internal (auto-created by Kafka Streams):**
- `KSTREAM-KEY-SELECT-0000000015-repartition` — from `selectKey()` in velocity chain
- `current-velocity-repartition`, `velocity-windows-repartition` — windowed aggregation boundaries
- `current-velocity-changelog`, `customerProfiles-STATE-STORE-changelog`, `ml-predictions-store-changelog` — RocksDB backups for state recovery

---

## Tech Stack

**Java layer:**
- Java 21, Spring Boot
- Apache Kafka Streams
- Spring AI (Groq / Llama 3.1 8b instant)

**Python layer:**
- Python 3.12
- XGBoost 2.0 + scikit-learn + SHAP (ML inference + explainability)
- Polars 0.20 (feature engineering, 5–10× faster than Pandas)
- River 0.21 `SRPClassifier` (online learning, concept drift resistant)
- ChromaDB 0.4 (local persistent vector store)
- sentence-transformers 2.7 `all-MiniLM-L6-v2` (384-dim, local, free)
- MLflow (XGBoost training experiment tracking)
- confluent-kafka 2.3, Pydantic v2, structlog
- Streamlit 1.29 + Plotly (real-time dashboard)

**Infrastructure:**
- Docker Compose — Kafka KRaft broker + Kafka UI (port 8090)
- 3 partitions, 1 replica (development)

---

## Project Structure

```
streaming-fraud-intelligence/
├── src/main/java/com/agenticfraud/engine/
│   ├── streaming/FraudStreams.java          ← Kafka Streams topology (4 sub-topologies)
│   ├── services/AgentCoordinator.java       ← 10-LLM-call orchestration
│   ├── agents/                              ← 5 specialized LLM agents
│   ├── models/                              ← Transaction, StreamingContext, MLPrediction
│   └── controllers/FraudDetectionController.java
├── python-ml/
│   ├── consumers/inference_consumer.py      ← XGBoost + SHAP + RAG at inference time
│   ├── agents/
│   │   ├── feedback_embedder.py             ← ChromaDB embedding pipeline
│   │   └── rag_retriever.py                 ← cosine similarity retrieval
│   ├── training/
│   │   ├── train_xgboost.py                 ← XGBoost training + MLflow
│   │   └── online_learner.py                ← River SRPClassifier
│   ├── features/engineer.py                 ← 19 Polars features
│   ├── models/schemas.py                    ← Pydantic models
│   ├── monitoring/dashboard.py              ← Streamlit dashboard
│   └── config.py
├── docker-compose.yml                       ← KRaft Kafka + Kafka UI
└── setup-topics.sh
```

---

## Future Enhancements

- **Evidently drift detection** — monitor feature distribution on `enriched-transactions`, publish alerts to `model-health` topic
- **LangMem agent memory** — persistent agent-level knowledge across sessions
- **Pydantic AI structured outputs** — replace regex parsing in `AgenticFraudUtils.java`
- **LSTM sequence model** — TensorFlow temporal sequence scoring alongside XGBoost
- **Human analyst confirmation endpoint** — REST endpoint writing `actualFraud=true` to `analyst-feedback` for ground-truth RAG seeding
- **Kubernetes deployment** — Strimzi operator for Kafka, HPA for Python inference pods

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Connect

- [LinkedIn](https://www.linkedin.com/in/siddhartha-devineni/)

**Built for the Kafka + AI community — give it a ⭐ if it helps you**