package com.agenticfraud.engine.models;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

public record FraudDecision(
    String transactionId,
    boolean isFraudulent,
    double confidenceScore,
    String primaryReason,
    String detailedExplanation,
    List<AgentInsight> agentInsights,
    Map<String, Object> riskFactors,
    LocalDateTime analyzedAt,
    String fraudPattern) {

  public static FraudDecision fraudulent(
      String transactionId,
      double confidence,
      String reason,
      String explanation,
      List<AgentInsight> insights) {
    return new FraudDecision(
        transactionId,
        true,
        confidence,
        reason,
        explanation,
        insights,
        Map.of(),
        LocalDateTime.now(),
        "unknown");
  }

  public static FraudDecision legitimate(
      String transactionId, double confidence, List<AgentInsight> insights) {
    return new FraudDecision(
        transactionId,
        false,
        confidence,
        "Transaction appears legitimate",
        "All agents agree this translation follows normal patterns",
        insights,
        Map.of(),
        LocalDateTime.now(),
        "legitimate");
  }

  public boolean isHighConfidence() {
    return confidenceScore >= 0.8;
  }

  public boolean requireManualReview() {
    return confidenceScore < 0.7 && confidenceScore > 0.3;
  }
}
