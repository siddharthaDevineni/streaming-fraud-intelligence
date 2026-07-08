from agents.base_agent import BaseFraudAgent


class BehaviorAnalyst(BaseFraudAgent):
    """
    Specializes in customer behavioral patterns and spending deviation.
    Weight 1.2x — behavior is a key fraud signal.
    Mirrors BehaviorAnalyst.java.
    """

    @property
    def agent_name(self) -> str:
        return "BEHAVIOR_ANALYST"

    @property
    def weight(self) -> float:
        return 1.2

    def _rag_instruction(self) -> str:
        return (
            "As a BEHAVIOR ANALYST: if similar confirmed cases show the same "
            "velocity + spending deviation pattern, weight this heavily in your "
            "RISK_SCORE. Confirmed historical cases with matching behavioral "
            "signatures are strong evidence of the same attack type."
        )

    def _build_analysis_prompt(
            self,
            transaction: dict,
            streaming_context: str,
    ) -> str:
        return f"""You are a fraud detection specialist analyzing customer 
            behavioral patterns.
            
            STREAMING INTELLIGENCE:
            {streaming_context}
            
            TRANSACTION:
            {self._format_transaction(transaction)}
            
            Analyze:
            1. Does transaction amount deviate from customer baseline?
            2. Does velocity pattern match normal customer behavior?
            3. Are merchant category and channel consistent with history?
            4. Do timing patterns suggest automated behavior?
            {self._build_rag_block(streaming_context)}{self._response_format()}"""

    def _build_collaboration_prompt(
            self,
            transaction: dict,
            question: str,
    ) -> str:
        return f"""You are a fraud detection specialist. A colleague asks:
                
            {question}
            
            TRANSACTION:
            {self._format_transaction(transaction)}
            
            Answer focusing on behavioral patterns.
            {self._response_format()}"""
