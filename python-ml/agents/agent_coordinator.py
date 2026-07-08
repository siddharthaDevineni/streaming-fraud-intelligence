import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import structlog
from langchain_groq import ChatGroq
from langsmith import traceable

from agents.base_agent import AgentInsight, BaseFraudAgent, parse_llm_response
from agents.behavior_analyst import BehaviorAnalyst
from agents.geographic_analyst import GeographicAnalyst
from agents.pattern_detector import PatternDetector
from agents.risk_assessor import RiskAssessor
from agents.temporal_analyst import TemporalAnalyst
from config import settings

logger = structlog.get_logger()

class AgentCoordinator:
    """
    Python port of AgentCoordinator.java.

    Orchestrates 10 LLM calls per transaction:
      Phase 1:  5 parallel specialized agents
      Phase 2a: 2 velocity collaboration calls (if high velocity)
      Phase 2b: 2 profile collaboration calls (if customer profile exists)
      Phase 2c: 1 consensus coordinator call
      Phase 3:  decision synthesis (no LLM call)

    Key difference from Java: RAG context is available at analysis time
    since Python owns both retrieval and reasoning — no KTable lag.
    """

    FRAUD_THRESHOLD = 0.6
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    DISAGREEMENT_THRESHOLD = 0.4

    def __init__(self):
        self.agents: list[BaseFraudAgent] = [
            BehaviorAnalyst(),
            PatternDetector(),
            RiskAssessor(),
            GeographicAnalyst(),
            TemporalAnalyst(),
        ]

        self.llm = ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key, temperature=0.1)
        self.executor = ThreadPoolExecutor(max_workers=5)
        logger.info("agent_coordinator_ready", agents=len(self.agents))

    @traceable(name="agent_coordinator", run_type="chain")
    def investigate(self, transaction: dict, streaming_context: str,
                    has_high_velocity: bool = False, has_customer_profile: bool = False,) -> dict:
        """
        Full multi-phase investigation — mirrors AgentCoordinator.investigateTransaction()
        Returns FraudDecision dict.
        """

        start_ms = time.time() * 1000

        logger.info("investigation_started", transactionId=transaction["transactionId"],
                    has_high_velocity=has_high_velocity, has_customer_profile=has_customer_profile)

        # Phase 1 - parallel specialized analysis
        phase1_insights = self._phase1_parallel(transaction, streaming_context)

        # Phase 2 - collaboration + consensus
        collaborative_insights = self._phase2_collaboration(transaction = transaction,
                                                            streaming_context = streaming_context,
                                                            phase1_insights = phase1_insights,
                                                            has_high_velocity = has_high_velocity,
                                                            has_customer_profile = has_customer_profile,)

        all_insights = phase1_insights + collaborative_insights

        # Phase 3 - decision synthesis
        decision = self._phase3_synthesize(transaction = transaction,
                                           streaming_context = streaming_context,
                                           all_insights = all_insights)

        duration_ms = int(time.time() * 1000 - start_ms)

        logger.info(
            "investigation_complete",
            transaction_id=transaction.get("transactionId"),
            is_fraudulent=decision["isFraudulent"],
            confidence=round(decision["confidenceScore"], 3),
            agents=len(all_insights),
            duration_ms=duration_ms,
        )

        return decision

    # ─── Phase 1 ──────────────────────────────────────────────────────────────

    @traceable(name="phase1-parallel-agents", run_type="chain")
    def _phase1_parallel(self, transaction: dict, streaming_context: str) -> list[AgentInsight]:
        """5 agents in parallel via ThreadPoolExecutor — mirrors CompletableFuture."""
        futures = {
            self.executor.submit(agent.analyze, transaction, streaming_context): agent.agent_name
            for agent in self.agents
        }

        insights = []
        for future in as_completed(futures):
            try:
                insights.append(future.result())
            except Exception as e:
                agent_name = futures[future]
                logger.error("agent_error", agent=agent_name, error=str(e))

        logger.info("phase1_complete", agents=len(insights))
        return insights

    # ─── Phase 2 ──────────────────────────────────────────────────────────────

    def _phase2_collaboration(self, transaction: dict, streaming_context: str,
                              phase1_insights: list[AgentInsight],
                              has_high_velocity: bool, has_customer_profile: bool,) -> list[AgentInsight]:
        """Collaboration + consensus — mirrors facilitateStreamingCollaboration()."""

        collaborative = []

        requires_collab = (self._has_disagreement(phase1_insights) or
                           has_high_velocity or
                           has_customer_profile)

        if not requires_collab:
            logger.info("standard_collaboration_sufficient")
            collaborative.append(self._build_consensus(transaction, streaming_context, phase1_insights))
            return collaborative

        logger.info("enhanced_collaboration_triggered")

        # Phase 2a + 2b — velocity and profile collaboration in parallel
        collab_futures = {}

        if has_high_velocity:
            logger.info("velocity_collaboration_triggered")
            collab_futures["velocity"] = self.executor.submit(
                self._velocity_collaboration, transaction
            )

        if has_customer_profile:
            logger.info("profile_collaboration_triggered")
            collab_futures["profile"] = self.executor.submit(
                self._profile_collaboration, transaction, streaming_context
            )

        for key, future in collab_futures.items():
            try:
                collaborative.extend(future.result())
            except Exception as e:
                logger.error(f"{key}_collaboration_failed", error=str(e))

        # Phase 2c — consensus coordinator (after both collaboration rounds)
        collaborative.append(
            self._build_consensus(transaction, streaming_context, phase1_insights)
        )

        return collaborative

    @traceable(name="velocity-collaboration", run_type="chain")
    def _velocity_collaboration(self, transaction: dict) -> list[AgentInsight]:
        """PatternDetector + TemporalAnalyst debate the velocity question."""

        velocity_count = transaction.get("velocityCount", 0)
        question = (
            f"High velocity detected ({velocity_count} transactions). "
            f"Does this align with automated attack patterns?"
        )

        pattern_agent = next(agent for agent in self.agents if agent.agent_name == "PATTERN_DETECTOR")
        temporal_agent = next(agent for agent in self.agents if agent.agent_name == "TEMPORAL_ANALYST")

        futures = {
            self.executor.submit(pattern_agent.collaborate, transaction, question),
            self.executor.submit(temporal_agent.collaborate, transaction, question),
        }
        return [future.result() for future in as_completed(futures)]

    @traceable(name="profile-collaboration", run_type="chain")
    def _profile_collaboration(
            self,
            transaction: dict,
            streaming_context: str,
    ) -> list[AgentInsight]:
        """BehaviorAnalyst + RiskAssessor debate customer profile question."""
        avg_amount = transaction.get("customerAvgAmount", 0)
        risk_level = transaction.get("customerRiskLevel", "UNKNOWN")
        question = (
            f"Customer profile shows ${avg_amount:.0f} average transactions, "
            f"{risk_level} risk level. How does this affect your analysis?"
        )

        behavior_agent = next(
            a for a in self.agents if a.agent_name == "BEHAVIOR_ANALYST"
        )
        risk_agent = next(
            a for a in self.agents if a.agent_name == "RISK_ASSESSOR"
        )

        futures = {
            self.executor.submit(behavior_agent.collaborate, transaction, question),
            self.executor.submit(risk_agent.collaborate, transaction, question),
        }
        return [f.result() for f in as_completed(futures)]

    @traceable(name="consensus-coordinator", run_type="llm")
    def _build_consensus(
            self,
            transaction: dict,
            streaming_context: str,
            insights: list[AgentInsight],
    ) -> AgentInsight:
        """STREAMING_CONSENSUS_COORDINATOR — mirrors buildStreamingConsensus()."""
        agent_summary = "\n".join(
            f"{i.agent_name} (Risk: {i.risk_score:.2f}): {i.reasoning[:200]}"
            for i in insights
        )

        prompt = f"""You are the lead fraud investigator with access to real-time streaming intelligence and historical fraud case context.

        STREAMING CONTEXT:
        {streaming_context}
        
        TRANSACTION:
        {self._format_transaction(transaction)}
        
        AGENT FINDINGS:
        {agent_summary}
        
        Based on streaming intelligence, RAG historical context, and agent analyses,
        provide final consensus:
        - If SIMILAR CONFIRMED FRAUD CASES appear above, how closely do they 
          match this transaction? High similarity (>70%) to confirmed fraud is 
          the strongest evidence available — your RISK_SCORE should be higher 
          when confirmed similar cases exist than when no historical context 
          is available.
        - How does streaming velocity and customer profile context affect risk?
        - What is the combined weight of all agent findings?
        - Your RISK_SCORE should reflect ALL available evidence: agent analyses,
          streaming context, AND historical case similarity.
        
        You MUST respond in EXACTLY this format. Each field on its OWN LINE. No exceptions:
        RISK_SCORE: [number between 0.0 and 1.0]
        REASONING: [your analysis in one paragraph]
        RECOMMENDATION: [FRAUD_ALERT or HUMAN_REVIEW or APPROVE]
        PATTERN: [card_testing or vpn_bot_fraud or account_takeover or general_fraud]
        
        Rules:
        - Each of the 4 fields above MUST be on a separate line starting with the field name
        - PATTERN must be the LAST line
        - For PATTERN, reason from current transaction features: high velocity (3+ txns) 
          + bot device + rapid fire = card_testing, regardless of historical case labels"""

        response = self.llm.invoke(prompt)
        insight = parse_llm_response(
            response.content,
            agent_name="STREAMING_CONSENSUS_COORDINATOR",
            weight=0.8,
        )

        return AgentInsight(
            agent_name="STREAMING_CONSENSUS_COORDINATOR",
            risk_score=insight.risk_score,
            confidence=insight.risk_score,
            reasoning=insight.reasoning,
            recommendation=insight.recommendation,
            weight=0.8,
            pattern=insight.pattern
        )

    # ─── Phase 3 ──────────────────────────────────────────────────────────────

    def _phase3_synthesize(
            self,
            transaction: dict,
            streaming_context: str,
            all_insights: list[AgentInsight],
    ) -> dict:
        """
        Final decision synthesis — mirrors synthesizeStreamingIntelligentDecision().
        No additional LLM call.
        """
        base_risk = self._weighted_risk_score(all_insights)
        streaming_bonus = self._streaming_bonus(transaction)
        final_risk = min(1.0, base_risk + streaming_bonus)

        is_fraudulent = final_risk >= self.FRAUD_THRESHOLD

        # Use the consensus orchestrator's risk score as confidence —
        # it is the LLM's own assessment of overall certainty after
        # reading all agent insights + RAG context + streaming intelligence.
        # This is what the LLM decided.
        consensus_insight = next(
            (i for i in all_insights
             if i.agent_name == "STREAMING_CONSENSUS_COORDINATOR"),
            None
        )
        if consensus_insight:
            confidence = round(consensus_insight.risk_score, 3)
            fraud_pattern = consensus_insight.pattern  # ← LLM-decided pattern
            logger.info(
                "consensus_pattern_extracted",
                pattern=fraud_pattern,
                confidence=confidence,
                transaction_id=transaction.get("transactionId"),)
        else:
            # Fallback only if consensus somehow missing
            confidence = self._calculate_confidence(all_insights, is_fraudulent)
            fraud_pattern = "unknown"

        logger.info(
            "decision_synthesized",
            base_risk=round(base_risk, 4),
            streaming_bonus=round(streaming_bonus, 4),
            final_risk=round(final_risk, 4),
            is_fraudulent=is_fraudulent,
            confidence=round(confidence, 3),
            agents=len(all_insights),
        )

        return {
            "transactionId": transaction.get("transactionId"),
            "isFraudulent": is_fraudulent,
            "confidenceScore": confidence,
            "fraudPattern": fraud_pattern,
            "finalRiskScore": final_risk,
            "agentCount": len(all_insights),
            "explanation": self._build_explanation(
                transaction, streaming_context, all_insights, final_risk
            ),
            "agentInsights": [
                {
                    "agentName": i.agent_name,
                    "riskScore": i.risk_score,
                    "reasoning": i.reasoning,
                }
                for i in all_insights
            ],
        }

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _weighted_risk_score(self, insights: list[AgentInsight]) -> float:
        """Mirrors calculateWeightedRiskScore() — weight by agent specialization."""
        if not insights:
            return 0.5
        total_score = sum(i.risk_score * i.weight for i in insights)
        total_weight = sum(i.weight for i in insights)
        return total_score / total_weight if total_weight > 0 else 0.5

    def _streaming_bonus(self, transaction: dict) -> float:
        """Mirrors calculateStreamingIntelligenceBonus()."""
        bonus = 0.0
        if transaction.get("hasHighVelocity"):
            bonus += 0.25
        if transaction.get("isAmountUnusual"):
            bonus += 0.20
        if transaction.get("customerRiskLevel") == "HIGH":
            bonus += 0.10
        return bonus

    def _calculate_confidence(
            self,
            insights: list[AgentInsight],
            is_fraudulent: bool,
    ) -> float:
        """Mirrors calculateConfidence() — agreement ratio → confidence tier."""
        if not insights:
            return 0.3
        agreeing = sum(
            1 for i in insights if i.indicates_fraud() == is_fraudulent
        )
        ratio = agreeing / len(insights)
        logger.info(
            "confidence_calculation",
            agreeing=agreeing,
            total=len(insights),
            ratio=round(ratio, 3),
        )
        if ratio >= 0.8:
            return 0.9
        if ratio >= 0.6:
            return 0.7
        if ratio >= 0.4:
            return 0.5
        return 0.3

    def _has_disagreement(self, insights: list[AgentInsight]) -> bool:
        """Triggers collaboration when agents disagree significantly."""
        if len(insights) < 2:
            return False
        scores = [i.risk_score for i in insights]
        return (max(scores) - min(scores)) > self.DISAGREEMENT_THRESHOLD

    def _format_transaction(self, transaction: dict) -> str:
        return (
            f"ID: {transaction.get('transactionId')}\n"
            f"Amount: ${transaction.get('amount')} {transaction.get('currency')}\n"
            f"Merchant: {transaction.get('merchantId')} "
            f"({transaction.get('merchantCategory')})\n"
            f"Location: {transaction.get('location')}\n"
            f"Device: {transaction.get('metadata', {}).get('deviceId')}\n"
            f"Rapid fire: {transaction.get('metadata', {}).get('rapidFire')}"
        )

    def _build_explanation(
            self,
            transaction: dict,
            streaming_context: str,
            insights: list[AgentInsight],
            final_risk: float,
    ) -> str:
        lines = ["AI AGENTS ENHANCED WITH STREAMING INTELLIGENCE\n"]
        lines.append(f"TRANSACTION:\n{self._format_transaction(transaction)}\n")
        lines.append(f"STREAMING CONTEXT:\n{streaming_context}\n")
        lines.append("AGENT ANALYSIS:")
        for i in insights:
            lines.append(
                f"- {i.agent_name} (Risk: {i.risk_score*100:.1f}%): {i.reasoning[:150]}"
            )
        lines.append(
            f"\nFINAL RISK SCORE: {final_risk*100:.1f}%\n"
            f"DECISION: {'FRAUD DETECTED' if final_risk >= self.FRAUD_THRESHOLD else 'LEGITIMATE'}"
        )
        return "\n".join(lines)
