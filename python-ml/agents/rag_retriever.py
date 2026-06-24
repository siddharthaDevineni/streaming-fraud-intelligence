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
      → injected into MLPrediction.ragContext field
      → Java reads ragContext from ml-predictions topic
      → injected into each agent's buildStreamingAnalysisPrompt()
"""

import chromadb
import structlog
import time
from config import settings
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger()

# How many similar cases to retrieve per transaction
DEFAULT_N_RESULTS = 3

# Minimum similarity score to include a case (cosine similarity: 0=unrelated, 1=identical)
# Cases below this threshold are not useful context for agents
MIN_SIMILARITY_SCORE = 0.50


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
                name=settings.chroma_collection_fraud
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
                metadata={"hnsw:space": "cosine"},
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

        # Velocity-based metadata filter — ensures high-velocity queries only
        # retrieve high-velocity confirmed cases, preventing vpn_bot_fraud
        # (velocity=1) from dominating results for card_testing (velocity=15)
        # scenarios where the two patterns share 16 of 19 features
        is_high_velocity = bool(features.get("is_high_velocity", 0))
        velocity_filter = {"isHighVelocity": 1} if is_high_velocity else {"isHighVelocity": 0}

        if customer_id:
            # Try customer-specific first — if CUST-001 was flagged before,
            # that history is directly relevant
            customer_results = self._query_chromadb(
                query_embedding=query_embedding,
                n_results=min(2, n_results),
                where={"customerId": customer_id, "isHighVelocity": int(is_high_velocity)},
            )
        else:
            customer_results = []

        # General cases — velocity-filtered to prevent pattern bleed
        general_n = n_results - len(customer_results)
        general_results = self._query_chromadb(
            query_embedding=query_embedding,
            n_results=general_n + 2,  # fetch extra, deduplicate below
            where=velocity_filter,
        )

        # Merge: customer-specific cases first, then general cases
        # Deduplicate by transactionId
        seen_ids = {r["metadata"]["transactionId"] for r in customer_results}
        deduped_general = [
            r for r in general_results
            if r["metadata"]["transactionId"] not in seen_ids
        ]

        combined = customer_results + deduped_general
        results = combined[:n_results]

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

    def _query_chromadb(
            self,
            query_embedding: list,
            n_results: int,
            where: dict = None,
    ) -> list[dict]:
        """Execute ChromaDB query and normalize results into clean dicts."""
        if n_results <= 0:
            return []

        # ChromaDB raises if n_results > collection size
        actual_n = min(n_results, self.collection.count())
        if actual_n == 0:
            return []

        try:
            kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": actual_n,
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                kwargs["where"] = where

            results = self.collection.query(**kwargs)

        except Exception as e:
            # If velocity-filtered query fails (e.g. no matching cases yet),
            # fall back to unfiltered query so retrieval is never blocked
            if where:
                logger.debug(
                    "chromadb_filtered_query_failed_falling_back",
                    filter=where,
                    error=str(e),
                )
                try:
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=actual_n,
                        include=["documents", "metadatas", "distances"],
                    )
                except Exception as e2:
                    logger.error("chromadb_query_failed", error=str(e2))
                    return []
            else:
                logger.error("chromadb_query_failed", error=str(e))
                return []

        # ChromaDB returns distances (lower = more similar for cosine)
        # Convert to similarity score (higher = more similar): similarity = 1 - distance
        normalized = []
        for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
        ):
            normalized.append({
                "text": doc,
                "similarity": round(1 - distance, 4),
                "metadata": metadata,
            })

        return normalized

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

            lines.append(
                f"Case {i} (similarity: {similarity_pct}%, "
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
        This is serialized to JSON and sent via ml-predictions topic to Java.
        Java reads it from StreamingContext and injects into agent prompts.
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
