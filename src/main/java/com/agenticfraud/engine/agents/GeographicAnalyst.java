package com.agenticfraud.engine.agents;

import com.agenticfraud.engine.models.CustomerProfile;
import com.agenticfraud.engine.models.StreamingContext;
import com.agenticfraud.engine.models.Transaction;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.stereotype.Component;

@Component
public class GeographicAnalyst extends AbstractFraudAgent {

  protected GeographicAnalyst(ChatModel chatModel) {
    super(chatModel);
  }

  @Override
  protected String buildAnalysisPrompt(Transaction transaction) {
    return String.format(
        """
                You are an expert Geographic Risk Analyst for fraud detection.

                Analyze the geographic risk factors for this transaction:
                %s

                Evaluate:
                1. Location consistency with customer's typical patterns
                2. Geographic impossibility (physical travel time between transactions)
                3. High-risk geographic regions or areas
                4. Cross-border transaction risks
                5. Location spoofing indicators
                6. Regional fraud trends and hotspots

                Consider:
                - Travel patterns and feasibility
                - Regional crime statistics
                - Known fraud rings operating in the area
                - Geopolitical risk factors
                - Local merchant legitimacy

                Provide your analysis in this format:
                RISK_SCORE: [0.0-1.0]
                REASONING: [Geographic risk factors analysis]
                RECOMMENDATION: [Location-based fraud prevention action]

                Be specific about geographic inconsistencies or patterns you identify.
                """,
        transaction.toAnalysisText());
  }

  @Override
  protected String buildStreamingAnalysisPrompt(Transaction transaction, StreamingContext context) {
    // Geographic Analyst focuses on LOCATION CONTEXT and TRAVEL PATTERNS
    StringBuilder prompt = new StringBuilder();

    prompt.append("You are an expert Geographic Risk Analyst for fraud detection.\n\n");

    // Emphasize location intelligence (Geographic Analyst's specialty)
    prompt.append("STREAMING INTELLIGENCE - LOCATION RISK ANALYSIS:\n");

    // Customer location baseline
    if (context.customerProfile() != null) {
      CustomerProfile profile = context.customerProfile();

      prompt.append("CUSTOMER LOCATION BASELINE:\n");
      prompt.append(String.format("- Primary location: %s\n", profile.primaryLocation()));
      prompt.append(String.format("- Risk tier: %s\n", profile.riskLevel()));

      // Current transaction location
      String currentLocation = transaction.location();
      prompt.append(String.format("- Current transaction: %s\n", currentLocation));

      // Check location consistency
      if (profile.isTypicalLocation(currentLocation)) {
        prompt.append("Location matches customer baseline (low risk)\n");
      } else {
        prompt.append("LOCATION ANOMALY: Transaction from unusual location!\n");
        prompt.append(
            String.format(
                "   Expected: %s | Actual: %s\n", profile.primaryLocation(), currentLocation));
      }
      prompt.append("\n");
    } else {
      prompt.append("No location baseline available\n");
      prompt.append("Analyzing transaction location in isolation\n\n");
    }

    // Velocity context for geographic impossibility check
    if (context.hasHighVelocity()) {
      prompt.append("GEOGRAPHIC IMPOSSIBILITY CHECK:\n");
      prompt.append(
          String.format(
              "%d transactions in 5 minutes from potentially different locations\n",
              context.recentTransactionsCount()));
      prompt.append("CRITICAL QUESTION: Is rapid travel between locations physically possible?\n");
      prompt.append("Red flags:\n");
      prompt.append("- Transactions from different cities/states in short time = IMPOSSIBLE\n");
      prompt.append("- Same city but rapid succession = Possible but suspicious\n");
      prompt.append("- Unknown/masked locations = Potential VPN/proxy usage\n");
      prompt.append("- Location jumps = Card cloning or account takeover\n\n");
    }

    // Transaction location details
    prompt.append("TRANSACTION LOCATION DETAILS:\n");
    prompt.append(transaction.toAnalysisText());
    prompt.append("\n");
    prompt.append(String.format("- Location: %s\n", transaction.location()));
    prompt.append(String.format("- Merchant: %s\n", transaction.merchantId()));
    prompt.append(String.format("- Category: %s\n", transaction.merchantCategory()));

    // Check for suspicious location indicators
    String location = transaction.location().toLowerCase();
    if (location.contains("unknown") || location.contains("unavailable")) {
      prompt.append("ALERT: Location information unavailable (possible VPN/proxy)\n");
    }
    prompt.append("\n");

    // rag context to the prompt
    if (context.ragContext() != null && !context.ragContext().isEmpty()) {
      Object promptContext = context.ragContext().get("promptContext");
      if (promptContext != null && !promptContext.toString().isBlank()) {
        prompt.append(promptContext.toString());
        prompt.append("\n");
        prompt.append(
            "As a GEOGRAPHIC ANALYST: if similar cases show the same unknown location pattern, this is confirmed geo-anomaly.\n\n");
      }
    }

    // Geographic risk assessment framework
    prompt.append("As a GEOGRAPHIC ANALYST, evaluate:\n");
    prompt.append("1. LOCATION CONSISTENCY: Does location match customer baseline?\n");
    prompt.append("2. GEOGRAPHIC IMPOSSIBILITY: Can customer physically be at this location?\n");
    prompt.append("   - Check: Could they travel from last transaction location in time?\n");
    prompt.append("3. REGIONAL RISK: Is this location a known fraud hotspot?\n");
    prompt.append("4. LOCATION SPOOFING: Signs of VPN/proxy/location masking?\n");
    prompt.append("5. CROSS-BORDER RISK: International transaction risks?\n\n");

    prompt.append("Geographic Risk Factors:\n");
    prompt.append("- Location jump + high velocity = Account takeover (HIGH RISK)\n");
    prompt.append("- Unknown location + unusual merchant = VPN fraud (MEDIUM RISK)\n");
    prompt.append("- Consistent location + normal velocity = Legitimate (LOW RISK)\n\n");

    prompt.append("Provide your analysis in this format:\n");
    prompt.append("RISK_SCORE: [0.0-1.0]\n");
    prompt.append("REASONING: [Geographic risk analysis with travel feasibility assessment]\n");
    prompt.append("RECOMMENDATION: [Location-based fraud prevention action]\n");

    return prompt.toString();
  }

  @Override
  protected void initializeKnowledge() {
    knowledgeBase.put("high_risk_regions", "known_fraud_hotspots");
    knowledgeBase.put("travel_speed_limits", "max_reasonable_travel_mph");
    knowledgeBase.put("cross_border_flags", "international_transaction_risks");
    knowledgeBase.put("location_spoofing", "vpn_proxy_indicators");
  }

  @Override
  public String getSpecialization() {
    return "Geographic Risk Analysis";
  }

  @Override
  public String getAgentId() {
    return "GEOGRAPHIC_ANALYST";
  }
}
