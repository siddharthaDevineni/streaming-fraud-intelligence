from agents.base_agent import BaseFraudAgent


class TemporalAnalyst(BaseFraudAgent):
    """
    Specializes in timing patterns and bot behavior detection.
    Weight 1.0x — standard weight.
    Mirrors TemporalAnalyst.java.
    """

    @property
    def agent_name(self) -> str:
        return "TEMPORAL_ANALYST"

    @property
    def weight(self) -> float:
        return 1.0

    def _rag_instruction(self) -> str:
        return (
            "As a TEMPORAL ANALYST: if similar confirmed cases show the same "
            "rapid-fire bot timing pattern (sub-second intervals, rapidFire=true), "
            "this is confirmed automated attack behaviour — significantly increase "
            "your RISK_SCORE. Bot timing patterns in confirmed cases are "
            "definitive evidence of scripted attacks."
        )

    def _build_analysis_prompt(self, transaction: dict, streaming_context: str) -> str:
        return f"""You are a temporal fraud analyst specializing in timing patterns and automated attack detection.

            STREAMING INTELLIGENCE:
            {streaming_context}
            
            TRANSACTION:
            {self._format_transaction(transaction)}
            
            Analyze:
            1. Are sub-second transaction intervals consistent with bot automation?
            2. Does transaction hour match customer's typical activity window?
            3. Does the rapid-fire flag combined with velocity indicate scripted attack?
            4. Is the timing pattern consistent with human or automated behavior?
            {self._build_rag_block(streaming_context)}{self._response_format()}"""

    def _build_collaboration_prompt(self, transaction: dict, question: str) -> str:
        return f"""You are a temporal fraud analyst. A colleague asks:

            {question}
            
            TRANSACTION:
            {self._format_transaction(transaction)}
            
            Answer focusing on timing patterns and automated attack indicators.
            {self._response_format()}"""