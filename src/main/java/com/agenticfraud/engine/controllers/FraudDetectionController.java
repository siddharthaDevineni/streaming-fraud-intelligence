package com.agenticfraud.engine.controllers;

import com.agenticfraud.engine.models.FraudDecision;
import com.agenticfraud.engine.models.StreamingContext;
import com.agenticfraud.engine.models.Transaction;
import com.agenticfraud.engine.services.AgentCoordinator;
import jakarta.validation.Valid;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/fraud-detection")
@CrossOrigin(origins = "*")
public class FraudDetectionController {

  private static final Logger logger = LoggerFactory.getLogger(FraudDetectionController.class);

  private final AgentCoordinator agentCoordinator;

  public FraudDetectionController(AgentCoordinator agentCoordinator) {
    this.agentCoordinator = agentCoordinator;
  }

  /**
   * Analyzes a given transaction using streaming-intelligent AI agents to determine if it is
   * potentially fraudulent. Even REST API calls benefit from streaming intelligence architecture
   *
   * @param transaction the transaction to be analyzed. Must be a valid {@link Transaction} object.
   * @return a {@link ResponseEntity} containing a {@link FraudDecision} that encapsulates the fraud
   *     analysis result, including whether the transaction is fraudulent, a confidence score, and
   *     additional analysis details.
   */
  @PostMapping("/analyze")
  public ResponseEntity<FraudDecision> analyzeTransaction(
      @Valid @RequestBody Transaction transaction) {

    logger.info(
        "New streaming-intelligent fraud analysis request for transaction: {}",
        transaction.transactionId());

    try {

      StreamingContext restApiContext = createRestApiStreamingContext(transaction);

      FraudDecision decision = agentCoordinator.investigateTransaction(transaction, restApiContext);

      logger.info(
          "Streaming-intelligent analysis complete: {} (confidence: {:.2f})",
          decision.isFraudulent() ? "FRAUD" : "LEGITIMATE",
          decision.confidenceScore());

      return ResponseEntity.ok(decision);
    } catch (Exception e) {
      logger.error(
          "Error in streaming-intelligent analysis {}: {}",
          transaction.transactionId(),
          e.getMessage());

      FraudDecision errorDecision =
          FraudDecision.fraudulent(
              transaction.transactionId(),
              0.5,
              "Technical Error in Streaming Analysis",
              "Analysis failed due to technical error: " + e.getMessage(),
              List.of());

      return ResponseEntity.internalServerError().body(errorDecision);
    }
  }

  /**
   * Create a streaming context for the REST API calls
   *
   * @param transaction transaction
   * @return StreamingContext
   */
  private StreamingContext createRestApiStreamingContext(Transaction transaction) {
    return new StreamingContext(
        null, null, "REST API call - Single transaction analysis", null, null);
  }

  /**
   * Get information about the streaming-intelligent AI agents
   *
   * @return Map of agent information
   */
  @GetMapping("/agents/info")
  public ResponseEntity<Map<String, Object>> getAgentsInfo() {

    Map<String, Object> agentInfo =
        Map.of(
            "totalAgents",
            5,
            "architecture",
            "Streaming-Intelligent AI",
            "agents",
            Map.of(
                "BEHAVIOR_ANALYST", "Analyzes customer behavior with velocity intelligence",
                "PATTERN_DETECTOR", "Identifies attack patterns with streaming context",
                "RISK_ASSESSOR", "Calculates risk with customer profile intelligence",
                "GEOGRAPHIC_ANALYST", "Evaluates location risks with streaming data",
                "TEMPORAL_ANALYST", "Analyzes timing patterns with velocity context"),
            "streamingCapabilities",
            List.of(
                "Real-time velocity intelligence",
                "Customer profile streaming context",
                "AI-enhanced pattern detection",
                "Streaming-intelligent decision synthesis",
                "Dynamic risk adjustment with streaming data"),
            "uniqueValue",
            "Kafka streaming expertise enhances AI intelligence",
            "version",
            "1.0.0");

    return ResponseEntity.ok(agentInfo);
  }

  @GetMapping("/health")
  public ResponseEntity<Map<String, Object>> healthCheck() {

    try {
      Map<String, Object> health =
          Map.of(
              "status", "UP",
              "architecture", "Streaming-Intelligent AI",
              "agentCoordinator", "STREAMING_INTELLIGENT_MODE",
              "aiSystem", "OPERATIONAL",
              "streamingIntelligence", "ACTIVE",
              "timestamp", java.time.LocalDateTime.now());

      return ResponseEntity.ok(health);
    } catch (Exception e) {
      Map<String, Object> health =
          Map.of(
              "status",
              "DOWN",
              "error",
              e.getMessage(),
              "architecture",
              "Streaming-Intelligent AI",
              "timestamp",
              java.time.LocalDateTime.now());
      return ResponseEntity.status(503).body(health);
    }
  }
}
