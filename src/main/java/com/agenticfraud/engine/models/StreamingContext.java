package com.agenticfraud.engine.models;

import java.util.Map;

/**
 * StreamingContext
 *
 * @param recentTransactionsCount recent transactions count in the last 5 minutes
 * @param customerProfile customer profile
 * @param contextSummary context summary
 */
public record StreamingContext(
    Long recentTransactionsCount, // Transactions in the last 5 minutes
    CustomerProfile customerProfile,
    String contextSummary,
    // from Python ml-predictions topic
    Double mlFraudScore,
    Map<String, Object> ragContext) {

  public boolean hasHighVelocity() {
    return recentTransactionsCount != null && recentTransactionsCount > 3;
  }

  public String getAIContext() {
    StringBuilder context = new StringBuilder();

    if (hasHighVelocity()) {
      context.append(
          String.format(
              "HIGH VELOCITY: %d transactions in the last 5 minutes", recentTransactionsCount));
    }

    if (customerProfile != null) {
      context.append(
          String.format(
              "Customer baseline: $%.2f avg, %s risk.",
              customerProfile.averageTransactionAmount().doubleValue(),
              customerProfile.riskLevel()));
    }

    // ML score injected into agent context
    if (mlFraudScore != null) {
      context.append(String.format(
              "ML MODEL PRE-SCREEN: XGBoost fraud score = %.1f%%. ",
              mlFraudScore * 100));
      if (mlFraudScore > 0.8) {
        context.append("ML strongly indicates fraud. ");
      } else if (mlFraudScore < 0.3) {
        context.append("ML indicates likely legitimate. ");
      }
    }

    // RAG context — similar past confirmed fraud cases
    if (ragContext != null && !ragContext.isEmpty()) {
      Object promptContext = ragContext.get("promptContext");
      Object casesFound = ragContext.get("casesFound");
      if (promptContext != null && !promptContext.toString().isBlank()) {
        context.append(String.format(
                "\n\n%s similar confirmed case(s) retrieved:\n%s",
                casesFound, promptContext));
      }
    }

    return context.toString();
  }
}
