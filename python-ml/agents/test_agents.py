from agents.behavior_analyst import BehaviorAnalyst
from agents.pattern_detector import PatternDetector
from agents.risk_assessor import RiskAssessor
from agents.geographic_analyst import GeographicAnalyst
from agents.temporal_analyst import TemporalAnalyst
from agents.agent_coordinator import AgentCoordinator

agents = [
    BehaviorAnalyst(),
    PatternDetector(),
    RiskAssessor(),
    GeographicAnalyst(),
    TemporalAnalyst(),
]

txn = {
    'transactionId': 'TEST-001',
    'amount': 30,
    'currency': 'USD',
    'merchantId': 'MERCHANT-SUSPICIOUS-1',
    'merchantCategory': 'ONLINE',
    'location': 'Unknown Location',
    'metadata': {'deviceId': 'BOT-DEVICE-1', 'channel': 'ONLINE', 'rapidFire': True}
}
context = 'HIGH VELOCITY: 15 transactions in 5 minutes. Customer baseline: \$244 avg, HIGH risk. XGBoost fraud score = 99.8%.'

for agent in agents:
    insight = agent.analyze(txn, context)
    print(f'{agent.agent_name}: risk={insight.risk_score}')

# Test AgentCoordinator:

coordinator = AgentCoordinator()

txn = {
    'transactionId': 'TEST-001',
    'amount': 30,
    'currency': 'USD',
    'merchantId': 'MERCHANT-SUSPICIOUS-1',
    'merchantCategory': 'ONLINE',
    'location': 'Unknown Location',
    'velocityCount': 15,
    'hasHighVelocity': True,
    'isAmountUnusual': True,
    'customerRiskLevel': 'HIGH',
    'customerAvgAmount': 244,
    'metadata': {
        'deviceId': 'BOT-DEVICE-1',
        'channel': 'ONLINE',
        'rapidFire': True
    }
}

context = '''HIGH VELOCITY: 15 transactions in 5 minutes
Customer baseline: $244 avg, HIGH risk.
XGBoost fraud score = 99.8%. ML strongly indicates fraud.
3 similar confirmed cases retrieved at 82% similarity — all card_testing.'''

decision = coordinator.investigate(
    transaction=txn,
    streaming_context=context,
    has_high_velocity=True,
    has_customer_profile=True,
)

print(f'Is fraudulent: {decision["isFraudulent"]}')
print(f'Confidence: {decision["confidenceScore"]}')
print(f'Final risk: {decision["finalRiskScore"]}')
print(f'Agents used: {decision["agentCount"]}')