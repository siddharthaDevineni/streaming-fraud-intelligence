package com.agenticfraud.engine.controllers;

import com.fasterxml.jackson.databind.ObjectMapper;
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
 * curl -X POST http://localhost:8080/api/investigation/chat \                                                                                                            -H "Content-Type: application/json" \
 *   -d '{"question": "How you doing?"}' \
 *   | python3 -m json.tool
 *   % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
 *                                  Dload  Upload   Total   Spent    Left  Speed
 * 100   964    0   934  100    30   2283     73 --:--:-- --:--:-- --:--:--  2356
 * {
 *     "response": "I'm doing well, thanks for asking. I'm ready to assist with any questions you have about the fraud detection system. \n\nTo confirm, I'll be answering questions based solely on the provided system architecture, without making any assumptions or inventing details not mentioned. If a question asks about something not described in the architecture, I'll let you know that it's outside the scope of the information provided. \n\nPlease go ahead and ask your question, and I'll do my best to provide an accurate and clear explanation based on the architecture.",
 *     "timestamp": "2026-07-05T16:52:28.187262",
 *     "systemCapabilities": [
 *         "Real-time Kafka Streams enrichment (Java)",
 *         "XGBoost + SHAP ML inference (Python)",
 *         "ChromaDB RAG historical case retrieval (Python)",
 *         "10 LLM calls per transaction \u2014 5 parallel + 4 collaborative + 1 consensus",
 *         "River SRPClassifier online learning (Python)",
 *         "LangSmith full pipeline observability"
 *     ]
 * }
 */
@RestController
@RequestMapping("/api/investigation")
@CrossOrigin(origins = "*")
public class ConversationalController {

  private static final Logger logger =
          LoggerFactory.getLogger(ConversationalController.class);

  private static final String PYTHON_BASE_URL = "http://localhost:8000";

  private final RestTemplate restTemplate;
  private final ObjectMapper objectMapper;

  public ConversationalController() {
    this.restTemplate = new RestTemplate();
    this.objectMapper = new ObjectMapper();
  }

  @PostMapping("/chat")
  public ResponseEntity<Map> chatWithSystem(
          @RequestBody Map<String, String> request) {

    String question = request.getOrDefault("question", "");
    logger.info("Chat question: {} — delegating to Python", question);

    try {
      String jsonBody = objectMapper.writeValueAsString(request);

      HttpHeaders headers = new HttpHeaders();
      headers.setContentType(MediaType.APPLICATION_JSON);
      headers.setAccept(List.of(MediaType.APPLICATION_JSON));

      HttpEntity<String> entity = new HttpEntity<>(jsonBody, headers);

      ResponseEntity<Map> response = restTemplate.exchange(
              PYTHON_BASE_URL + "/investigation/chat",
              HttpMethod.POST,
              entity,
              Map.class
      );

      return ResponseEntity.ok(response.getBody());

    } catch (RestClientException e) {
      logger.error("Chat request failed: {}", e.getMessage());
      return ResponseEntity.status(503).body(Map.of(
              "response", "Chat service temporarily unavailable.",
              "error", e.getMessage(),
              "timestamp", LocalDateTime.now().toString()));
    } catch (Exception e) {
      logger.error("Serialization error: {}", e.getMessage());
      return ResponseEntity.status(500).build();
    }
  }
}