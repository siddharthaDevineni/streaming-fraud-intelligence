package com.agenticfraud.engine.agents;

import com.agenticfraud.engine.models.StreamingContext;
import com.agenticfraud.engine.models.Transaction;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.stereotype.Component;

@Component
public class TemporalAnalyst extends AbstractFraudAgent {

  protected TemporalAnalyst(ChatModel chatModel) {
    super(chatModel);
  }

  @Override
  protected String buildAnalysisPrompt(Transaction transaction) {
    return String.format(
        """
                You are an expert Temporal Pattern Analyst specializing in fraud detection.

                Analyze the timing patterns and temporal risks for this transaction:
                %s

                Examine:
                1. Transaction time vs. customer's typical active hours
                2. Day of week and seasonal patterns
                3. Velocity patterns (frequency and timing of recent transactions)
                4. Holiday and weekend behavior anomalies
                5. Time zone inconsistencies with location
                6. Rapid-fire transaction sequences indicating automation

                Consider temporal fraud indicators:
                - Off-hours activity (middle of night transactions)
                - Unnatural transaction timing (too precise, too rapid)
                - Time zone manipulation attempts
                - Batch processing patterns indicating bot activity

                Provide your analysis in this format:
                RISK_SCORE: [0.0-1.0]
                REASONING: [Temporal patten analysis and anomalies]
                RECOMMENDATION: [Time-based risk mitigations]

                Focus on timing inconsistencies or suspicious temporal patterns.
                """,
        transaction.toAnalysisText());
  }

  @Override
  protected String buildStreamingAnalysisPrompt(Transaction transaction, StreamingContext context) {
    // Temporal Analyst focuses on TIMING VELOCITY in a streaming context
    StringBuilder prompt = new StringBuilder();

    prompt.append(
        "You are an expert Temporal Pattern Analyst specializing in fraud detection.\n\n");

    // Emphasizing timing velocity, which is Temporal Analyst's specialty
    prompt.append("STREAMING INTELLIGENCE - TEMPORAL VELOCITY ANALYSIS:\n");
    if (context.hasHighVelocity()) {
      prompt.append(
          String.format(
              "RAPID-FIRE PATTERN: %d transactions in 5-minute window\n",
              context.recentTransactionsCount()));
      prompt.append("Analyze timing intervals:\n");
      prompt.append("- Sub-second intervals = Bot activity\n");
      prompt.append("- Regular intervals = Automated script\n");
      prompt.append("- Burst pattern = Card testing attack\n");
    } else {
      prompt.append("Normal timing pattern\n");
    }
    prompt.append("\n");

    // Transaction timing details
    prompt.append("TRANSACTION TIMING:\n");
    prompt.append(transaction.toAnalysisText());
    prompt.append(String.format("\n- Hour: %d:00\n", transaction.timestamp().getHour()));
    prompt.append(String.format("- Day: %s\n", transaction.timestamp().getDayOfWeek()));
    prompt.append("\n");

    if (context.customerProfile() != null) {
      prompt.append("CUSTOMER TIMING BASELINE:\n");
      prompt.append(String.format("- Risk level: %s\n", context.customerProfile().riskLevel()));
      prompt.append("\n");
    }

    // rag context to the prompt
    if (context.ragContext() != null && !context.ragContext().isEmpty()) {
      Object promptContext = context.ragContext().get("promptContext");
      if (promptContext != null && !promptContext.toString().isBlank()) {
        prompt.append(promptContext.toString());
        prompt.append("\n");
        prompt.append(
            "As a TEMPORAL ANALYST: if similar cases occurred at the same hour, note the timing signature.\n\n");
      }
    }

    prompt.append("As a TEMPORAL ANALYST, focus on:\n");
    prompt.append("1. Does the VELOCITY indicate automated/bot activity?\n");
    prompt.append("2. Are transaction intervals suspicious (too fast/regular)?\n");
    prompt.append("3. Is the timing consistent with normal human behavior?\n");
    prompt.append("4. Do timing patterns match known attack vectors?\n\n");

    prompt.append("Provide your analysis in this format:\n");
    prompt.append("RISK_SCORE: [0.0-1.0]\n");
    prompt.append("REASONING: [Temporal velocity analysis]\n");
    prompt.append("RECOMMENDATION: [Time-based fraud prevention action]\n");

    return prompt.toString();
  }

  @Override
  protected void initializeKnowledge() {
    knowledgeBase.put("normal_hours", "6am-11pm_local_time");
    knowledgeBase.put("velocity_thresholds", "max_transactions_per_hour");
    knowledgeBase.put("automation_indicators", "sub_second_intervals");
    knowledgeBase.put("timezone_risks", "location_time_mismatches");
  }

  @Override
  public String getSpecialization() {
    return "Temporal Pattern Analysis";
  }

  @Override
  public String getAgentId() {
    return "TEMPORAL_ANALYST";
  }
}
