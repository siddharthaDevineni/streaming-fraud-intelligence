import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import structlog
from config import settings
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq

logger = structlog.get_logger()

@dataclass
class AgentInsight:
    agent_name: str
    risk_score: float
    confidence: float
    reasoning: str
    recommendation: str
    weight: float = 1.0
    pattern: str = "unknown"

    def indicates_fraud(self) -> bool:
        return self.risk_score >= 0.6

def parse_llm_response(text: str, agent_name: str = "unknown", weight: float = 1.0) -> AgentInsight:
    """
    Shared response parser used by both BaseFraudAgent and AgentCoordinator.
    Extracts RISK_SCORE, REASONING, RECOMMENDATION, PATTERN from LLM output.
    """
    import re

    def extract_float(t, pattern):
        m = re.search(pattern, t, re.IGNORECASE)
        return float(m.group(1).strip()) if m else 0.5

    def extract_text(t, pattern):
        m = re.search(pattern, t, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        return m.group(1).strip() if m else ""

    risk_score = extract_float(text, r"RISK_SCORE:\s*([0-9.]+)")
    risk_score = min(1.0, max(0.0, risk_score))
    reasoning = extract_text(text, r"REASONING:\s*(.+?)(?=\nRECOMMENDATION:|RECOMMENDATION:|PATTERN:|$)")
    recommendation = extract_text(text, r"RECOMMENDATION:\s*([^\r\n]+)")
    pattern = extract_text(text,
                           r"PATTERN:\s*(card_testing|vpn_bot_fraud|account_takeover|general_fraud)"
                           )

    return AgentInsight(
        agent_name=agent_name,
        risk_score=risk_score,
        confidence=risk_score,
        reasoning=reasoning,
        recommendation=recommendation,
        weight=weight,
        pattern=pattern if pattern else "unknown",
    )


class BaseFraudAgent(ABC):
    """
    Abstract base class for all fraud detection agents.
    Mirrors AbstractFraudAgent.java — each agent has a specialization,
    weight, and two prompt types (analysis + collaboration).
    """

    def __init__(self):
        self.llm = ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key, temperature=0.1)

    @property
    @abstractmethod
    def agent_name(self) -> str:
        pass

    @property
    @abstractmethod
    def weight(self) -> float:
        pass

    @abstractmethod
    def _build_analysis_prompt(
            self,
            transaction: dict,
            streaming_context: str,
    ) -> str:
        pass

    # each agent overrides with its own RAG weighting instruction
    def _rag_instruction(self) -> str:
        """
        Per-agent instruction for reasoning about RAG historical context.
        Mirrors Java's per-agent RAG weighting in buildStreamingAnalysisPrompt().
        Only injected into the prompt when confirmed cases are available.
        Override in each concrete agent class.
        """
        return (
            "If similar confirmed fraud cases are shown above, weight them "
            "heavily in your RISK_SCORE — they are validated historical evidence."
        )

    def _build_rag_block(self, streaming_context: str) -> str:
        """
        Returns the agent-specific RAG instruction only when confirmed
        historical cases are actually present in the streaming context.
        Returns empty string when no cases are available — avoids confusing
        the agent with an instruction about cases that do not exist.
        """
        if "SIMILAR CONFIRMED FRAUD CASES FROM HISTORY" in streaming_context:
            return f"\n{self._rag_instruction()}\n"
        return ""

    @abstractmethod
    def _build_collaboration_prompt(
            self,
            transaction: dict,
            question: str,
    ) -> str:
        pass

    def analyze(self, transaction: dict, streaming_context: str) -> AgentInsight:
        """Phase 1 - individual streaming-enhanced analysis."""
        prompt_text = self._build_analysis_prompt(transaction, streaming_context)
        response: AIMessage = self.llm.invoke(prompt_text)
        insight = self._parse_response(response.content)
        logger.info(f"{self.agent_name}_completed", risk=round(insight.risk_score, 2),
                    confidence=round(insight.confidence, 2),
                    transaction_id=transaction.get("transactionId"), )
        return insight

    def collaborate(self, transaction: dict, question: str) -> AgentInsight:
        """Phase 2 - collaborative analysis responding to a question."""
        prompt_text = self._build_collaboration_prompt(transaction, question)
        response: AIMessage = self.llm.invoke(prompt_text)
        return self._parse_response(response.content)

    def _parse_response(self, response: str) -> AgentInsight:
        insight = parse_llm_response(response, self.agent_name, self.weight)
        return insight

    def _extract_float(self, text: str, pattern: str) -> float:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                return float(match.group(1).strip())
            except ValueError:
                pass
        return 0.5  # safe default

    def _extract_text(self, text: str, pattern: str) -> str:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        return match.group(1).strip() if match else ""

    def _format_transaction(self, transaction: dict) -> str:
        return (
            f"Transaction ID: {transaction.get('transactionId')}\n"
            f"Amount: ${transaction.get('amount')} {transaction.get('currency')}\n"
            f"Merchant: {transaction.get('merchantId')} ({transaction.get('merchantCategory')})\n"
            f"Location: {transaction.get('location')}\n"
            f"Device: {transaction.get('metadata', {}).get('deviceId', 'unknown')}\n"
            f"Channel: {transaction.get('metadata', {}).get('channel', 'unknown')}\n"
            f"Rapid fire: {transaction.get('metadata', {}).get('rapidFire', False)}\n"
        )

    def _response_format(self) -> str:
        return (
            "\n\nRespond in exactly this format:\n"
            "RISK_SCORE: [0.0-1.0]\n"
            "REASONING: [your analysis]\n"
            "RECOMMENDATION: [FRAUD_ALERT|HUMAN_REVIEW|APPROVE]"
        )
