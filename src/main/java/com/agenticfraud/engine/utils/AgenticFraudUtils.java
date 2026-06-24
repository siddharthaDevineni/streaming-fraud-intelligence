package com.agenticfraud.engine.utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class AgenticFraudUtils {

  protected static final Logger logger = LoggerFactory.getLogger(AgenticFraudUtils.class);

  private AgenticFraudUtils() {}

    /**
     * Extract the risk score from the AI analysis string.
     * @param analysis The analysis string.
     * @return The risk score.
     */
  public static double extractRiskScore(String analysis) {
    try {
      // Look for RISK_SCORE: pattern
      if (analysis.contains("RISK_SCORE:")) {
        String scorePart = analysis.substring(analysis.indexOf("RISK_SCORE:") + 11);
        String scoreStr = scorePart.split("\\n")[0].trim().replaceAll("[^0-9.]", "");
        return Double.parseDouble(scoreStr);
      }

      // Fallback: analyze sentiment and keywords for risk estimation
      String lower = analysis.toLowerCase();
      if (lower.contains("high risk")
          || lower.contains("fraudulent")
          || lower.contains("suspicious")) {
        return 0.8;
      } else if (lower.contains("medium risk")
          || lower.contains("unusual")
          || lower.contains("concerning")) {
        return 0.6;
      } else if (lower.contains("low risk")
          || lower.contains("normal")
          || lower.contains("legitimate")) {
        return 0.2;
      }

      return 0.5; // Default moderate risk
    } catch (Exception e) {
      logger.warn("Could not extract risk score from analysis: {}", e.getMessage());
      return 0.5;
    }
  }

  public static String extractReasoning(String analysis) {
    if (analysis.contains("REASONING:")) {
      String reasoningPart = analysis.substring(analysis.indexOf("REASONING:") + 10);
      return reasoningPart.split("RECOMMENDATION:")[0].trim();
    }
    return analysis.length() > 200 ? analysis.substring(0, 200) + "..." : analysis;
  }

  public static String extractRecommendation(String analysis) {
    if (analysis.contains("RECOMMENDATION:")) {
      return analysis.substring(analysis.indexOf("RECOMMENDATION:") + 15).trim();
    }
    return "Standard fraud review recommended";
  }
}
