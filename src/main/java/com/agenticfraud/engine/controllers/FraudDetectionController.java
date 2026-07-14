package com.agenticfraud.engine.controllers;

import com.agenticfraud.engine.models.FraudDecision;
import com.agenticfraud.engine.models.Transaction;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import jakarta.validation.Valid;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestTemplate;

/**
 * curl -X POST http://localhost:8080/api/fraud-detection/analyze   -H "Content-Type: application/json"   -d '{
 *     "transactionId": "CURL-TEST-002",
 *     "customerId": "CUST-001",
 *     "amount": 30,
 *     "currency": "USD",
 *     "merchantId": "MERCHANT-SUSPICIOUS-7",
 *     "merchantCategory": "ONLINE",
 *     "location": "Unknown Location",
 *     "timestamp": "2026-07-05T10:00:00",
 *     "metadata": {
 *       "deviceId": "BOT-DEVICE-1",
 *       "channel": "ONLINE",
 *       "rapidFire": true
 *     }
 *   }' | python3 -m json.tool
 */
@RestController
@RequestMapping("/api/fraud-detection")
@CrossOrigin(origins = "*")
public class FraudDetectionController {

  private static final Logger logger =
          LoggerFactory.getLogger(FraudDetectionController.class);

  private static final String PYTHON_BASE_URL =
          System.getenv().getOrDefault("PYTHON_ML_SERVICE_URL", "http://localhost:8000");

  private final RestTemplate restTemplate;
  private final ObjectMapper objectMapper;

  public FraudDetectionController() {
    this.restTemplate = new RestTemplate();
    this.objectMapper = new ObjectMapper()
            .registerModule(new JavaTimeModule()).configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES,
            false);
  }

  @PostMapping("/analyze")
  public ResponseEntity<FraudDecision> analyzeTransaction(
          @Valid @RequestBody Transaction transaction) {

    logger.info(
            "REST API fraud analysis request for {} — delegating to Python",
            transaction.transactionId());

    try {
      // Serialize manually — guarantees body bytes are written correctly
      String jsonBody = objectMapper.writeValueAsString(
              Map.of("transaction", transaction)
      );

      HttpHeaders headers = new HttpHeaders();
      headers.setContentType(MediaType.APPLICATION_JSON);
      headers.setAccept(List.of(MediaType.APPLICATION_JSON));

      HttpEntity<String> entity = new HttpEntity<>(jsonBody, headers);

      ResponseEntity<String> response = restTemplate.exchange(
              PYTHON_BASE_URL + "/analyze",
              HttpMethod.POST,
              entity,
              String.class
      );
      String rawBody = response.getBody();

      // Deserialize manually — gives us full control over field mapping
      FraudDecision decision = objectMapper.readValue(rawBody, FraudDecision.class);

      logger.info(
              "Analysis complete for {}: {} (confidence: {}%)",
              transaction.transactionId(),
              decision.isFraudulent() ? "FRAUD" : "LEGITIMATE",
              Math.round(decision.confidenceScore() * 100));

      return ResponseEntity.ok(decision);

    } catch (RestClientException e) {
      logger.error(
              "Python ML service unavailable for {}: {}",
              transaction.transactionId(),
              e.getMessage());

      FraudDecision errorDecision = FraudDecision.fraudulent(
              transaction.transactionId(),
              0.5,
              "Python ML service unavailable — manual review required",
              "Analysis failed: " + e.getMessage(),
              List.of());

      return ResponseEntity.status(503).body(errorDecision);

    } catch (Exception e) {
      logger.error("Serialization error for {}: {}", transaction.transactionId(), e.getMessage());
      return ResponseEntity.status(500).build();
    }
  }

  @GetMapping("/agents/info")
  public ResponseEntity<Map<String, Object>> getAgentsInfo() {
    return ResponseEntity.ok(Map.of(
            "architecture", "Kafka Streams (Java) + LangChain Agents (Python)",
            "totalLlmCallsPerTransaction", 10,
            "phases", Map.of(
                    "phase1", "5 specialized agents in parallel (ThreadPoolExecutor)",
                    "phase2a", "PatternDetector + TemporalAnalyst velocity collaboration",
                    "phase2b", "BehaviorAnalyst + RiskAssessor profile collaboration",
                    "phase2c", "STREAMING_CONSENSUS_ORCHESTRATOR"),
            "agents", Map.of(
                    "BEHAVIOR_ANALYST",   "velocity + spending deviation (1.2x weight)",
                    "PATTERN_DETECTOR",   "known attack signatures (1.3x weight)",
                    "RISK_ASSESSOR",      "financial risk + customer profile (1.1x weight)",
                    "GEOGRAPHIC_ANALYST", "location anomaly + VPN detection (1.0x weight)",
                    "TEMPORAL_ANALYST",   "timing patterns + bot indicators (1.0x weight)"),
            "javaSide", List.of(
                    "Kafka Streams topology (4 sub-topologies)",
                    "Velocity windows — RocksDB state store",
                    "Customer profile KTable join",
                    "Fraud decision routing (split/branch)",
                    "EXACTLY_ONCE_V2 processing guarantee"),
            "pythonSide", List.of(
                    "XGBoost inference (19 Polars features, SHAP explainability)",
                    "ChromaDB RAG (sentence-transformers, top-3 case retrieval)",
                    "LangChain agent orchestration (Groq API)",
                    "River SRPClassifier online learning",
                    "LangSmith full pipeline observability")));
  }

  @GetMapping("/health")
  public ResponseEntity<Map<String, Object>> healthCheck() {
    String pythonStatus = checkPythonHealth();
    Map<String, Object> health = Map.of(
            "status", "DOWN".equals(pythonStatus) ? "DEGRADED" : "UP",
            "javaKafkaStreams", "UP",
            "pythonMlService", pythonStatus,
            "timestamp", LocalDateTime.now());

    int httpStatus = "DOWN".equals(pythonStatus) ? 503 : 200;
    return ResponseEntity.status(httpStatus).body(health);
  }

  private String checkPythonHealth() {
    try {
      restTemplate.getForEntity(PYTHON_BASE_URL + "/health", String.class);
      return "UP";
    } catch (Exception e) {
      logger.warn("Python health check failed: {}", e.getMessage());
      return "DOWN";
    }
  }
}