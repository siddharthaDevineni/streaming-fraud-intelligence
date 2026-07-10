"""
RAG Retriever
=============
Queries ChromaDB for confirmed fraud cases similar to a new transaction.
Called at inference time from inference_consumer.py BEFORE publishing
the MLPrediction — so agents receive historical context in their prompts.

Flow:
  new transaction features
      → build query text (same format as stored cases)
      → embed with sentence-transformers
      → ChromaDB cosine similarity search
      → return top N similar confirmed cases
      → injected into streaming_context string in inference_consumer.py
      → passed directly to Python AgentCoordinator
      → injected into every agent prompt at analysis time (no KTable lag)
"""

import time

import chromadb
import structlog
from config import settings
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger()

# How many similar cases to retrieve per transaction
# Set to 5 to ensure enough candidates after deduplication —
# if all cases belong to CUST-001, then customer-specific
# and general queries overlap; fetching 5 ensures 3 useful cases
# survive similarity filtering
DEFAULT_N_RESULTS = 5

# Minimum cosine similarity to include a case as agent context
# 0.40 is appropriate for all-MiniLM-L6-v2 (384-dim) where:
# - Same fraud pattern, different velocity: ~0.75-0.85 similarity
# - Different fraud pattern, same customer: ~0.50-0.65 similarity
# - Unrelated transactions: <0.40 similarity
# In practice with numpy re-embedding, all card_testing cases
# score 0.75+ so this threshold mainly excludes genuinely different patterns
MIN_SIMILARITY_SCORE = 0.40


class RAGRetriever:

    def __init__(self, embedding_model: SentenceTransformer = None):
        """
        embedding_model: pass the already-loaded model from inference_consumer
        to avoid loading it twice. If None, loads its own instance.
        """
        if embedding_model is not None:
            self.embedding_model = embedding_model
            logger.info("rag_retriever_using_shared_embedding_model")
        else:
            logger.info("rag_retriever_loading_embedding_model")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.collection = self._init_chromadb()
        self.retrieval_count = 0
        self.cache_hit_count = 0

        logger.info(
            "rag_retriever_ready",
            cases_available=self.collection.count(),
        )

    def _init_chromadb(self):
        """Connect to the same ChromaDB instance as feedback_embedder."""
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir,
                                           settings=chromadb.Settings(anonymized_telemetry=False))
        try:
            collection = client.get_collection(
                name=settings.chroma_collection_fraud,
            )
            logger.info(
                "chromadb_collection_loaded",
                cases=collection.count(),
            )
            return collection
        except Exception:
            # Collection doesn't exist yet — embedder hasn't run yet
            # Create empty collection so retriever doesn't crash
            logger.warning(
                "chromadb_collection_empty",
                hint="Run feedback_embedder.py and process some transactions first",
            )
            return client.get_or_create_collection(
                name=settings.chroma_collection_fraud,
                metadata={
                    "hnsw:space": "cosine",  # cosine similarity for text
                    "hnsw:construction_ef": 100,
                    "hnsw:M": 16,
                },
            )

    # ── Query text builder ────────────────────────────────────────────────────

    def _build_query_text(self, features: dict) -> str:
        """
        Build query text in the SAME FORMAT as feedback_embedder._build_fraud_case_text.

        Critical: if the query format differs from the stored format,
        cosine similarity will be low even for identical fraud patterns.
        The vocabulary must match exactly.

        We omit outcome/confirmation fields (those are only in stored cases)
        and focus on the observable transaction features.
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

        return f"""velocity: {velocity} transactions in 5 minutes {"(high velocity attack)" if is_high_velocity else ""}
        amount ratio: {amount_ratio:.2f}x customer baseline {"(unusual amount)" if is_amount_unusual else ""}
        merchant: {merchant_type} {"(atypical category)" if not is_typical_category else ""}
        location: {location_status}
        device: {"bot device detected" if is_bot_device else "normal device"}
        rapid fire flag: {"yes" if is_rapid_fire else "no"}
        transaction hour: {hour}:00
        customer risk tier: {customer_risk}"""

    # ── Core retrieval ────────────────────────────────────────────────────────
    def retrieve(
            self,
            features: dict,
            customer_id: str = None,
            n_results: int = DEFAULT_N_RESULTS,
    ) -> list[dict]:
        """
        Retrieve N most similar confirmed fraud cases from ChromaDB.

        Args:
            features: 19-feature dict from features/engineer.py
            customer_id: if provided, also retrieves customer-specific history
            n_results:   number of cases to retrieve

        Returns:
            List of dicts, each containing:
            - text:       the stored fraud case description
            - similarity: cosine similarity score (0-1, higher = more similar)
            - metadata:   pattern, confidence, date, customerId etc.
        """
        if self.collection.count() == 0:
            logger.debug("chromadb_empty_no_cases_yet")
            return []

        start_ms = time.time() * 1000

        query_text = self._build_query_text(features)

        query_embedding = self.embedding_model.encode(query_text).tolist()

        # Fetch top candidates without WHERE filter — ChromaDB 0.4.x has
        # known issues returning correct n_results when WHERE filter is combined
        # with HNSW index. Filter by customerId in Python instead.
        fetch_n = max(n_results * 4, 20)  # fetch enough to find n_results after filtering
        all_results = self._search_similar(
            query_embedding=query_embedding,
            n_results=fetch_n,
        )

        if customer_id:
            # Prioritise customer-specific cases — put them first
            customer_results = [
                r for r in all_results
                if r["metadata"].get("customerId") == customer_id
            ]
            other_results = [
                r for r in all_results
                if r["metadata"].get("customerId") != customer_id
            ]
            combined = customer_results + other_results
        else:
            combined = all_results

        results = combined[:n_results]

        # logger.info("pre_filter_similarities",
        #             count=len(combined),
        #             similarities=[round(r["similarity"], 3) for r in combined[:10]])

        # Filter out low-similarity results — not useful context for agents
        results = [r for r in results if r["similarity"] >= MIN_SIMILARITY_SCORE]

        latency_ms = int(time.time() * 1000 - start_ms)
        self.retrieval_count += 1

        logger.info(
            "rag_retrieval_complete",
            cases_found=len(results),
            customer_specific=len(customer_results),
            latency_ms=latency_ms,
            top_similarity=round(results[0]["similarity"], 3) if results else 0,
            top_pattern=results[0]["metadata"].get("fraudPattern") if results else None,
        )

        return results

    def _search_similar(
            self,
            query_embedding: list,
            n_results: int,
    ) -> list[dict]:
        """
        In-memory cosine similarity — re-embeds stored documents at query time.
        Bypasses ChromaDB HNSW index which has multi-process corruption issues.
        Safe for concurrent read/write from separate processes.
        """
        import numpy as np

        try:
            client = chromadb.PersistentClient(
                path=settings.chroma_persist_dir,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            collection = client.get_collection(
                name=settings.chroma_collection_fraud,
            )
            count = collection.count()
        except Exception:
            return []

        if count == 0:
            return []

        # Fetch documents and metadata — NOT embeddings (may be None in 0.6.x)
        all_docs = collection.get(
            include=["documents", "metadatas"],
            limit=count,
        )

        documents = all_docs["documents"]
        metadatas = all_docs["metadatas"]

        if not documents:
            return []

        # Re-embed stored documents for similarity comparison
        # Fast enough for <1000 docs — all-MiniLM-L6-v2 does batch in ~50ms
        stored_embeddings = self.embedding_model.encode(
            documents,
            batch_size=32,
            show_progress_bar=False,
        )

        query_vec = np.array(query_embedding)
        stored_vecs = np.array(stored_embeddings)

        # Cosine similarity
        stored_norms = np.linalg.norm(stored_vecs, axis=1, keepdims=True)
        query_norm = np.linalg.norm(query_vec)
        stored_normalized = stored_vecs / (stored_norms + 1e-10)
        query_normalized = query_vec / (query_norm + 1e-10)
        similarities = stored_normalized @ query_normalized

        # Top N by similarity
        top_indices = np.argsort(similarities)[::-1][:n_results]

        return [
            {
                "text": documents[idx],
                "similarity": round(float(similarities[idx]), 4),
                "metadata": metadatas[idx],
            }
            for idx in top_indices
        ]

    # ── Prompt injection ──────────────────────────────────────────────────────

    def format_for_agent_prompt(self, similar_cases: list[dict]) -> str:
        """
        Format retrieved cases into the text injected into agent prompts.

        This is what agents actually read. Format is designed to be:
        - Scannable: agents can quickly identify the most relevant case
        - Specific: exact numbers rather than vague descriptions
        - Actionable: includes what signals confirmed the fraud
        """
        if not similar_cases:
            return ""

        lines = ["SIMILAR CONFIRMED FRAUD CASES FROM HISTORY:"]
        lines.append("")

        for i, case in enumerate(similar_cases, 1):
            similarity_pct = int(case["similarity"] * 100)
            metadata = case["metadata"]
            pattern = metadata.get("fraudPattern", "unknown")
            confidence = metadata.get("confidence", 0)
            confirmed_by = metadata.get("confirmedBy", "system")
            embedded_at = metadata.get("embeddedAt", "")[:10]  # date only

            txn_id = metadata.get("transactionId", "unknown")
            txn_short = txn_id[-8:] if txn_id != "unknown" else "unknown"

            lines.append(
                f"Case {i} (similarity: {similarity_pct}%, "
                f"pattern: {pattern}, confidence: {confidence:.2f}, "
                f"txn: ...{txn_short}, "
                f"confirmed: {embedded_at}, by: {confirmed_by}):"
            )
            # Add the stored case text, indented
            for line in case["text"].split("\n"):
                lines.append(f"  {line}")
            lines.append("")

        lines.append(
            "Use these cases as precedent. "
            "Higher similarity = more relevant to current transaction."
        )

        return "\n".join(lines)

    def format_for_ml_prediction(self, similar_cases: list[dict]) -> dict:
        """
        Format retrieved cases as a dict for the MLPrediction.ragContext field.
        This is serialized to JSON and sent via ml-predictions topic to the dashboard and feedback embedder.
        """
        if not similar_cases:
            return {
                "casesFound": 0,
                "cases": [],
                "promptContext": "",
            }

        return {
            "casesFound": len(similar_cases),
            "cases": [
                {
                    "similarity": case["similarity"],
                    "fraudPattern": case["metadata"].get("fraudPattern"),
                    "confidence": case["metadata"].get("confidence"),
                    "confirmedBy": case["metadata"].get("confirmedBy"),
                    "summary": case["text"][:200],  # truncated for Kafka payload
                }
                for case in similar_cases
            ],
            "promptContext": self.format_for_agent_prompt(similar_cases),
        }

    # ── Utility ───────────────────────────────────────────────────────────────

    def collection_stats(self) -> dict:
        """Return stats about the ChromaDB collection."""
        count = self.collection.count()
        return {
            "totalCases": count,
            "retrievalCount": self.retrieval_count,
            "collectionName": settings.chroma_collection_fraud,
            "persistDir": settings.chroma_persist_dir,
        }
