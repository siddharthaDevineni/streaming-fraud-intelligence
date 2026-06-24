package com.agenticfraud.engine.models;

public record EnrichedTransaction(
    Transaction transaction, CustomerProfile customerProfile, Long velocityCount) {

  public StreamingContext toStreamingContext() {
    String contextSummary = buildContextSummary();
    return new StreamingContext(
        velocityCount,
        customerProfile,
        contextSummary,
        null, // filled in by FraudStreams join
        null // filled in by FraudStreams join
        );
  }

  private String buildContextSummary() {
    StringBuilder summary = new StringBuilder("Streaming Context: ");

    if (velocityCount != null && velocityCount > 1) {
      summary.append(String.format("%d recent transactions", velocityCount));
      if (velocityCount > 3) {
        summary.append(" (HIGH VELOCITY)");
      }
      summary.append(", ");
    }
    if (customerProfile != null) {
      summary.append(
          String.format(
              "Customer: $%.0f avg, %s risk",
              customerProfile.averageTransactionAmount().doubleValue(),
              customerProfile.riskLevel()));

      if (customerProfile.isAmountUnusual(transaction.amount())) {
        summary.append(" (UNUSUAL AMOUNT)");
      }
    } else {
      summary.append("Real-time transaction analysis");
    }

    return summary.toString();
  }
}
