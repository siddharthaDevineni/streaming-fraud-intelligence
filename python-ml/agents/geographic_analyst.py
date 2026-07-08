from agents.base_agent import BaseFraudAgent


class GeographicAnalyst(BaseFraudAgent):
    """
    Specializes in location anomaly and VPN/proxy detection.
    Weight 1.0x — standard weight.
    Mirrors GeographicAnalyst.java.
    """

    @property
    def agent_name(self) -> str:
        return "GEOGRAPHIC_ANALYST"

    @property
    def weight(self) -> float:
        return 1.0

    def _build_analysis_prompt(self, transaction: dict, streaming_context: str) -> str:
        return f"""You are a geographic fraud analyst specializing in location anomaly detection and VPN/proxy identification.

            STREAMING INTELLIGENCE:
            {streaming_context}
            
            TRANSACTION:
            {self._format_transaction(transaction)}
            
            Analyze:
            1. Does transaction location match customer's primary location baseline?
            2. Does Unknown Location suggest VPN or proxy usage?
            3. Is geographic travel between transactions physically impossible?
            4. Does location pattern match known fraud hotspots?
            {self._build_rag_block(streaming_context)}{self._response_format()}"""

    def _build_collaboration_prompt(self, transaction: dict, question: str) -> str:
        return f"""You are a geographic fraud analyst. A colleague asks:

            {question}
            
            TRANSACTION:
            {self._format_transaction(transaction)}
            
            Answer focusing on location anomalies and VPN/proxy indicators.
            {self._response_format()}"""

    def _rag_instruction(self) -> str:
        return (
            "As a GEOGRAPHIC ANALYST: if similar confirmed cases show the same "
            "unknown/vpn location pattern, this is confirmed fraudulent "
            "geographic behaviour — weight heavily in your RISK_SCORE. "
            "Repeated unknown location + VPN across confirmed cases is a "
            "definitive fraud indicator."
        )