from agents.base_agent import BaseFraudAgent


class PatternDetector(BaseFraudAgent):
    """
    Specializes in known fraud attack signatures.
    Weight 1.3x — pattern matching is the strongest fraud signal.
    Mirrors PatternDetector.java.
    """

    @property
    def agent_name(self) -> str:
        return "PATTERN_DETECTOR"

    @property
    def weight(self) -> float:
        return 1.3

    def _rag_instruction(self) -> str:
        return (
            "As a PATTERN DETECTOR: if similar confirmed cases match the same "
            "attack signature (card_testing, vpn_bot_fraud, account_takeover), "
            "this is direct confirmation of a known pattern — significantly "
            "increase your RISK_SCORE. Pattern match to confirmed fraud is "
            "the strongest signal available."
        )

    def _build_analysis_prompt(self, transaction: dict, streaming_context: str) -> str:
        return f"""You are a fraud pattern detection specialist identifying known attack signatures.

            STREAMING INTELLIGENCE:
            {streaming_context}
            
            TRANSACTION:
            {self._format_transaction(transaction)}
            
            Analyze:
            1. Does velocity + amount pattern match card testing attack?
            2. Does merchant + location combination match known fraud vectors?
            3. Is the device ID consistent with bot/automated attack patterns?
            4. Does rapid-fire flag combined with small amounts indicate card probing?
            {self._build_rag_block(streaming_context)}{self._response_format()}"""

    def _build_collaboration_prompt(self, transaction: dict, question: str) -> str:
        return f"""You are a fraud pattern detection specialist. A colleague asks:

            {question}
            
            TRANSACTION:
            {self._format_transaction(transaction)}
            
            Answer focusing on known fraud attack patterns.
            {self._response_format()}"""