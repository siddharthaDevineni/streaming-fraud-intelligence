package com.agenticfraud.engine.agents;

import com.agenticfraud.engine.models.CustomerProfile;
import com.agenticfraud.engine.models.StreamingContext;
import com.agenticfraud.engine.models.Transaction;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.stereotype.Component;

@Component
public class RiskAssessor extends AbstractFraudAgent {

  protected RiskAssessor(ChatModel chatModel) {
    super(chatModel);
  }

  @Override
  protected String buildAnalysisPrompt(Transaction transaction) {
    return String.format(
        """
                You are an expert Financial Risk Assessor for fraud prevention.

                Evaluate the overall fraud risk for this transaction:
                %s

                Calculate risk based on:
                1. Transaction amount relative to account limits and history
                2. Merchant risk profile and category
                3. Geographic risk factors
                4. Time-based risk (off-hours, holidays)
                5. Channel risk (online vs. in-person)
                6. Currency and cross-border factors

                Consider both financial impact and probability:
                - High amount + high probability = critical risk
                - Low amount + high probability = moderate risk
                - High amount + low probability = monitoring needed

                Provide your assessment in this format:
                RISK_SCORE: [0.0-1.0]
                REASONING: [Comprehensive risk calculation methodology]
                RECOMMENDATION: [Risk mitigation action - approve, decline or additional verification]

                Provide specific risk factors and their weights in your analysis.
                """,
        transaction.toAnalysisText());
  }

  @Override
  protected String buildStreamingAnalysisPrompt(Transaction transaction, StreamingContext context) {
    // Risk Assessor focuses on FINANCIAL RISK with CUSTOMER BASELINE
    StringBuilder prompt = new StringBuilder();

    prompt.append("You are an expert Financial Risk Assessor for fraud prevention.\n\n");

    // Emphasize financial risk assessment (Risk Assessor's specialty)
    prompt.append("STREAMING INTELLIGENCE - FINANCIAL RISK ASSESSMENT:\n");

    // Customer profile baseline (most important for risk assessment)
    if (context.customerProfile() != null) {
      CustomerProfile profile = context.customerProfile();

      prompt.append("CUSTOMER FINANCIAL BASELINE:\n");
      prompt.append(
          String.format(
              "- Average transaction: $%.2f\n", profile.averageTransactionAmount().doubleValue()));
      prompt.append(
          String.format(
              "- Daily spending limit: $%.2f\n", profile.dailySpendingLimit().doubleValue()));
      prompt.append(String.format("- Risk tier: %s\n", profile.riskLevel()));

      // Calculate deviation from baseline
      double currentAmount = transaction.amount().doubleValue();
      double avgAmount = profile.averageTransactionAmount().doubleValue();
      double deviationMultiplier = currentAmount / avgAmount;

      prompt.append(String.format("- Current vs Average: %.1fx ", deviationMultiplier));
      if (deviationMultiplier > 3.0) {
        prompt.append("SIGNIFICANT DEVIATION (>3x average)\n");
      } else if (deviationMultiplier > 1.5) {
        prompt.append("Moderate deviation\n");
      } else {
        prompt.append("Within normal range\n");
      }

      // Check if unusual
      if (profile.isAmountUnusual(transaction.amount())) {
        prompt.append("ALERT: Amount flagged as unusual for this customer!\n");
      }

      // Check category consistency
      if (!profile.isTypicalCategory(transaction.merchantCategory())) {
        prompt.append(
            String.format(
                "Unusual category: %s (Typical: %s)\n",
                transaction.merchantCategory(),
                String.join(", ", profile.transactionCategories())));
      }

      prompt.append("\n");
    } else {
      prompt.append("LIMITED BASELINE: No customer profile available\n");
      prompt.append("Risk assessment based on transaction data only\n\n");
    }

    // Velocity context for risk multiplier
    if (context.hasHighVelocity()) {
      prompt.append("VELOCITY RISK MULTIPLIER:\n");
      prompt.append(
          String.format(
              "HIGH VELOCITY: %d transactions in 5 minutes\n", context.recentTransactionsCount()));
      prompt.append("Risk multiplier: 1.5x (rapid transaction activity)\n");
      prompt.append("Potential risk: Account takeover or card testing\n\n");
    }

    // Transaction risk details
    prompt.append("TRANSACTION RISK FACTORS:\n");
    prompt.append(transaction.toAnalysisText());
    prompt.append("\n");
    prompt.append(String.format("- Amount: $%.2f\n", transaction.amount().doubleValue()));
    prompt.append(
        String.format(
            "- Merchant: %s (%s)\n", transaction.merchantId(), transaction.merchantCategory()));
    prompt.append(
        String.format(
            "- Channel: %s\n", transaction.metadata().getOrDefault("channel", "UNKNOWN")));
    prompt.append("\n");

    // rag context to the prompt
    if (context.ragContext() != null && !context.ragContext().isEmpty()) {
      Object promptContext = context.ragContext().get("promptContext");
      if (promptContext != null && !promptContext.toString().isBlank()) {
        prompt.append(promptContext.toString());
        prompt.append("\n");
        prompt.append(
            "As a RISK ASSESSOR: confirmed similar cases are ground truth — weight your risk score upward.\n\n");
      }
    }

    // Risk assessment framework
    prompt.append("As a RISK ASSESSOR, calculate comprehensive risk:\n");
    prompt.append("1. FINANCIAL IMPACT: How does amount compare to baseline?\n");
    prompt.append("2. PROBABILITY: Given velocity + profile, what's fraud likelihood?\n");
    prompt.append("3. RISK TIER: Classify as LOW/MEDIUM/HIGH/CRITICAL\n");
    prompt.append("4. MERCHANT RISK: Is merchant category high-risk?\n");
    prompt.append("5. CUSTOMER RISK: Does customer's baseline increase/decrease risk?\n\n");

    prompt.append("Risk Calculation Formula:\n");
    prompt.append("Base Risk = Amount Deviation × Merchant Risk × Channel Risk\n");
    prompt.append("Final Risk = Base Risk × Velocity Multiplier × Customer Risk Tier\n\n");

    prompt.append("Provide your assessment in this format:\n");
    prompt.append("RISK_SCORE: [0.0-1.0]\n");
    prompt.append("REASONING: [Financial risk calculation with baseline comparison]\n");
    prompt.append("RECOMMENDATION: [Approve/Decline/Additional Verification with rationale]\n");

    return prompt.toString();
  }

  @Override
  protected void initializeKnowledge() {
    knowledgeBase.put("high_risk_merchants", "gambling,crypto,cash_advance");
    knowledgeBase.put("risk_multipliers", "international=1.5,off_hours=1.2");
    knowledgeBase.put("amount_thresholds", "low=100,medium=1000,high=5000");
    knowledgeBase.put("geographic_risk", "country_risk_scores");
  }

  @Override
  public String getSpecialization() {
    return "Financial Risk Assessment";
  }

  @Override
  public String getAgentId() {
    return "RISK_ASSESSOR";
  }
}
