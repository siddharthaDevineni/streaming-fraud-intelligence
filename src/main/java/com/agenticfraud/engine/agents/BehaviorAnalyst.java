package com.agenticfraud.engine.agents;

import com.agenticfraud.engine.models.CustomerProfile;
import com.agenticfraud.engine.models.StreamingContext;
import com.agenticfraud.engine.models.Transaction;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.stereotype.Component;

@Component
public class BehaviorAnalyst extends AbstractFraudAgent {

  public BehaviorAnalyst(ChatModel chatModel) {
    super(chatModel);
  }

  @Override
  protected String buildAnalysisPrompt(Transaction transaction) {
    return String.format(
        """
                You are an expert Customer Behavior Analyst specializing in fraud detection.

                Analyze this transaction for behavior anomalies:
                %s

                Focus on:
                1. Transaction timing patterns (unusual hours, frequency)
                2. Amount deviations from typical spending
                3. Location/merchant consistency with past behavior
                4. Payment method and channel analysis

                Consider normal variations vs. suspicious deviations.

                Provide your analysis in this format:
                RISK_SCORE: [0.0-1.0]
                REASONING: [Your detailed behavioral analysis]
                RECOMMENDATION: [Specific action based on behavioral patterns]

                Be thorough but concise. Focus on behavioral red flags or normal patterns.
                """,
        transaction.toAnalysisText());
  }

  @Override
  protected String buildStreamingAnalysisPrompt(Transaction transaction, StreamingContext context) {
    // Behavior Analyst focuses on VELOCITY PATTERNS in a streaming context
    StringBuilder prompt = new StringBuilder();

    prompt.append(
        "You are an expert Customer Behavior Analyst specializing in fraud detection.\n\n");

    // Emphasizing velocity intelligence, which is Behavior Analyst's specialty
    prompt.append("STREAMING INTELLIGENCE - VELOCITY ANALYSIS:\n");
    if (context.hasHighVelocity()) {
      prompt.append(
          String.format(
              "HIGH VELOCITY ALERT: %d transactions in the last 5 minutes\n",
              context.recentTransactionsCount()));
      prompt.append("This is HIGHLY UNUSUAL for normal customer behavior!\n");
    } else {
      prompt.append("Normal transaction velocity detected\n");
    }
    prompt.append("\n");

    // Customer baseline context
    if (context.customerProfile() != null) {
      CustomerProfile profile = context.customerProfile();
      prompt.append("CUSTOMER BEHAVIORAL BASELINE:\n");
      prompt.append(
          String.format(
              "- Average spending: $%.2f\n", profile.averageTransactionAmount().doubleValue()));
      prompt.append(String.format("- Risk level: %s\n", profile.riskLevel()));
      prompt.append(
          String.format(
              "- Typical categories: %s\n", String.join(", ", profile.transactionCategories())));
      prompt.append(String.format("- Primary location: %s\n\n", profile.primaryLocation()));
    }

    prompt.append("TRANSACTION TO ANALYZE:\n");
    prompt.append(transaction.toAnalysisText());
    prompt.append("\n\n");

    // rag context to the prompt
    if (context.ragContext() != null && !context.ragContext().isEmpty()) {
      Object promptContext = context.ragContext().get("promptContext");
      if (promptContext != null && !promptContext.toString().isBlank()) {
        prompt.append(promptContext.toString());
        prompt.append("\n");
        prompt.append(
            "As a BEHAVIOR ANALYST: if similar cases show same velocity+device pattern, weight heavily.\n\n");
      }
    }

    prompt.append("As a BEHAVIOR ANALYST, focus on:\n");
    prompt.append("1. How does the VELOCITY pattern affect behavioral risk?\n");
    prompt.append("2. Does spending amount deviate from customer baseline?\n");
    prompt.append("3. Are there behavioral red flags in frequency/timing?\n");
    prompt.append("4. Is this consistent with customer's normal behavior?\n\n");

    prompt.append("Provide your analysis in this format:\n");
    prompt.append("RISK_SCORE: [0.0-1.0]\n");
    prompt.append("REASONING: [Behavioral analysis with velocity context]\n");
    prompt.append("RECOMMENDATION: [Action based on behavioral patterns]\n");

    return prompt.toString();
  }

  @Override
  protected void initializeKnowledge() {
    knowledgeBase.put("normal_transaction_hours", "6-23");
    knowledgeBase.put("suspicious_frequency_threshold", "5_per_hour");
    knowledgeBase.put("amount_deviation_threshold", "3x_average");
    knowledgeBase.put("location_change_threshold", "500_miles");
  }

  @Override
  public String getSpecialization() {
    return "Customer Behavior Analysis";
  }

  @Override
  public String getAgentId() {
    return "BEHAVIOR_ANALYST";
  }
}
