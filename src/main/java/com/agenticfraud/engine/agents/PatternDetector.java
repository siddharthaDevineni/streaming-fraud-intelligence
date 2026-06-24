package com.agenticfraud.engine.agents;

import com.agenticfraud.engine.models.CustomerProfile;
import com.agenticfraud.engine.models.StreamingContext;
import com.agenticfraud.engine.models.Transaction;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.stereotype.Component;

@Component
public class PatternDetector extends AbstractFraudAgent {

  protected PatternDetector(ChatModel chatModel) {
    super(chatModel);
  }

  @Override
  protected String buildAnalysisPrompt(Transaction transaction) {
    return String.format(
        """
                You are an expert Fraud Pattern Detection specialist.

                Analyze this transaction for known fraud pattern:
                %s

                Look for these fraud indicators:
                1. Round number amounts (often used in testing stolen cards)
                2. Unusual merchant categories for the customer
                3. Geographic impossibility (rapid location changes)
                4. Time-based patterns (rapid successive transactions)
                5. Known fraud hotspot locations
                6. Merchant category switching patterns

                Cross-reference against known fraud attack vectors:
                - Card testing attacks
                - Account takeover patterns
                - Synthetic identity fraud
                - First-party fraud indicators

                Provide your analysis in this format:
                RISK_SCORE: [0.0-1.0]
                REASONING: [Specific patterns identified or ruled out]
                RECOMMENDATION: [Action based on pattern analysis]

                Be specific about which patterns you detect or explicitly rule out.
                """,
        transaction.toAnalysisText());
  }

  @Override
  protected String buildStreamingAnalysisPrompt(Transaction transaction, StreamingContext context) {
    // Pattern Detector focuses on ATTACK VECTORS in streaming context
    StringBuilder prompt = new StringBuilder();

    prompt.append("You are an expert Fraud Pattern Detection specialist.\n\n");

    // Emphasizing attack pattern detection, where it is Pattern Detector's specialty
    prompt.append("STREAMING INTELLIGENCE - ATTACK VECTOR ANALYSIS:\n");
    if (context.hasHighVelocity()) {
      prompt.append(
          String.format(
              "ATTACK PATTERN DETECTED: %d rapid transactions\n",
              context.recentTransactionsCount()));
      prompt.append("Known attack vectors this matches:\n");
      prompt.append("- Card Testing: Small amounts to test stolen cards\n");
      prompt.append("- Credential Stuffing: Rapid automated login attempts\n");
      prompt.append("- Account Takeover: Quick succession to drain account\n");
      prompt.append("- Bot Attack: Automated fraud script\n");
    } else {
      prompt.append("Single transaction - analyze for standalone fraud patterns\n");
    }
    prompt.append("\n");

    // Transaction pattern details
    prompt.append("TRANSACTION PATTERN:\n");
    prompt.append(transaction.toAnalysisText());
    prompt.append("\n");

    if (context.customerProfile() != null) {
      CustomerProfile profile = context.customerProfile();
      prompt.append("CUSTOMER PATTERN BASELINE:\n");
      prompt.append(
          String.format(
              "- Typical amount: $%.2f\n", profile.averageTransactionAmount().doubleValue()));
      prompt.append(
          String.format(
              "- Typical categories: %s\n", String.join(", ", profile.transactionCategories())));
      prompt.append("\n");
    }

    // rag context to the prompt
    if (context.ragContext() != null && !context.ragContext().isEmpty()) {
      Object promptContext = context.ragContext().get("promptContext");
      if (promptContext != null && !promptContext.toString().isBlank()) {
        prompt.append(promptContext.toString());
        prompt.append("\n");
        prompt.append(
            "As a PATTERN DETECTOR: use these cases to identify known attack signatures.\n\n");
      }
    }

    prompt.append("As a PATTERN DETECTOR, focus on:\n");
    prompt.append("1. Does the VELOCITY pattern match known attack vectors?\n");
    prompt.append("2. Are there card testing indicators (small amounts)?\n");
    prompt.append("3. Is this part of an automated fraud campaign?\n");
    prompt.append("4. Do patterns match fraud rings or bot activity?\n\n");

    prompt.append("Provide your analysis in this format:\n");
    prompt.append("RISK_SCORE: [0.0-1.0]\n");
    prompt.append("REASONING: [Attack pattern analysis]\n");
    prompt.append("RECOMMENDATION: [Pattern-based fraud prevention]\n");

    return prompt.toString();
  }

  @Override
  protected void initializeKnowledge() {
    knowledgeBase.put("round_amounts", "100,200,500,1000");
    knowledgeBase.put("testing_amounts", "1,5,10,25");
    knowledgeBase.put("fraud_hotspots", "high_risk_zip_codes");
    knowledgeBase.put("velocity_thresholds", "rapid_succession_patterns");
  }

  @Override
  public String getSpecialization() {
    return "Fraud Pattern Detection";
  }

  @Override
  public String getAgentId() {
    return "PATTERN_DETECTOR";
  }
}
