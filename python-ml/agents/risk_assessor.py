from agents.base_agent import BaseFraudAgent


class RiskAssessor(BaseFraudAgent):
    """
    Specializes in financial risk and customer profile analysis.
    Weight 1.1x — financial context is important but secondary to patterns.
    Mirrors RiskAssessor.java.
    """

    @property
    def agent_name(self) -> str:
        return "RISK_ASSESSOR"

    @property
    def weight(self) -> float:
        return 1.1

    def _rag_instruction(self) -> str:
        return (
            "As a RISK ASSESSOR: if similar confirmed cases show the same "
            "customer risk tier and financial profile (amount ratio, daily limit), "
            "treat this as validated financial risk precedent. High similarity "
            "to confirmed fraud cases should significantly increase your RISK_SCORE."
        )

    def _build_analysis_prompt(self, transaction: dict, streaming_context: str) -> str:
        return f"""You are a financial risk assessment specialist evaluating 
            transaction risk levels.
            
            STREAMING INTELLIGENCE:
            {streaming_context}
            
            TRANSACTION:
            {self._format_transaction(transaction)}
            
            Analyze:
            1. What is the financial impact relative to customer baseline?
            2. Does the customer risk tier increase or decrease concern?
            3. Does velocity multiplier significantly increase fraud probability?
            4. Is merchant risk category HIGH, MEDIUM, or LOW?
            {self._build_rag_block(streaming_context)}{self._response_format()}"""

    def _build_collaboration_prompt(self, transaction: dict, question: str) -> str:
        return f"""You are a financial risk assessment specialist. A colleague asks:

            {question}
            
            TRANSACTION:
            {self._format_transaction(transaction)}
            
            Answer focusing on financial risk and customer profile context.
            {self._response_format()}"""