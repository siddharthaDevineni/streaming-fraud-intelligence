package com.agenticfraud.engine.controllers;

import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/investigation")
@CrossOrigin(origins = "*")
public class ConversationalController {

  private static final Logger logger = LoggerFactory.getLogger(ConversationalController.class);

//  private final ChatModel chatModel;
//  private final AgentCoordinator agentCoordinator;
//
//  public ConversationalController(ChatModel chatModel, AgentCoordinator agentCoordinator) {
//    this.chatModel = chatModel;
//    this.agentCoordinator = agentCoordinator;
//  }

  /**
   * Chat with the fraud detection system
   *
   * @param request Request
   * @return Response
   */
  @PostMapping("/chat")
  public ResponseEntity<Map<String, Object>> chatWithSystem(
      @RequestBody Map<String, String> request) {

    String question = request.get("question");
    String transactionId = request.get("transactionId"); // Optional

    logger.info("New chat question: {}", question);

    try {
      String systemPrompt =
          """
                    You are an expert fraud investigation assistant with access to a multi-agent AI fraud detection system.

                    Our system uses 5 specialized AI agents:
                    - BehaviorAnalyst: Customer behavior patterns
                    - PatternDetector: Known fraud patterns
                    - RiskAssessor: Financial risk calculation
                    - GeographicAnalyst: Location-based risks
                    - TemporalAnalyst: Timing pattern analysis

                    The agents work together, debate findings, and reach consensus through weighted voting.

                    Use question: %s

                    Provide helpful, detailed responses about fraud detection, our AI system capabilities, or general fraud
                    prevention guidance. Be professional but accessible.

                    If asked about specific transactions, explain that you'd need transaction details to provide specific analysis.
                    """
              .formatted(question);

//      String response = chatModel.call(systemPrompt);

      Map<String, Object> chatResponse =
          Map.of(
              "response", "null",
              "timestamp", java.time.LocalDateTime.now(),
              "systemCapabilities",
                  List.of(
                      "Multi-agent fraud analysis",
                      "Explainable AI decision",
                      "Real-time transaction processing",
                      "Collaborative agent intelligence"));
      return ResponseEntity.ok(chatResponse);
    } catch (Exception e) {
      logger.error("Error processing chat request: {}", e.getMessage());

      Map<String, Object> chatResponse =
          Map.of(
              "response", "I'm sorry, I encountered a technical issue. Please try again.",
              "error", "Technical error occurred",
              "timestamp", java.time.LocalDateTime.now());

      return ResponseEntity.status(500).body(chatResponse);
    }
  }

  /**
   * Provides an explanation for the fraud decision made by the system regarding the specified
   * transaction.
   *
   * @param transactionId the unique identifier of the transaction whose decision needs to be
   *     explained.
   * @param context an optional map containing contextual details related to the transaction or user
   *     interactions. This can include additional metadata or parameters to refine the explanation
   *     process.
   * @return a ResponseEntity containing a map with the explanation details. The map may include the
   *     decision, reasoning, contributing factors, and any relevant metadata supporting the
   *     explanation.
   */
  @PostMapping("/explain/{transactionId}")
  public ResponseEntity<Map<String, Object>> explainDecision(
      @PathVariable String transactionId,
      @RequestBody(required = false) Map<String, Object> context) {

    logger.info("Explaining decision for transaction: {}", transactionId);

    try {

      String explanationPrompt =
          """
                    Explain this fraud detection decision in simple, conversational terms:

                    Transaction ID: %s

                    Our 5 AI fraud investigators analyzed this transaction:
                    - BehaviorAnalyst found behavioral patterns
                    - PatternDetector checked for known fraud indicators
                    - RiskAssessor calculated financial risk
                    - GeographicAnalyst evaluated location factors
                    - TemporalAnalyst examined timing patterns

                    Explain how these agents work together and what factors they consider.
                    Make it sound like you're explaining a real investigation team's work.
                    """
              .formatted(transactionId);

//      String explanation = chatModel.call(explanationPrompt);

      Map<String, Object> explanationResponse =
          Map.of(
              "transactionId",
              transactionId,
              "explanation",
              "explanation",
              "investigationProcess",
              List.of(
                  "Parallel agent analysis",
                  "Agent collaboration and debate",
                  "Consensus building",
                  "Final decision synthesis"),
              "timestamp",
              java.time.LocalDateTime.now());

      return ResponseEntity.ok(explanationResponse);
    } catch (Exception e) {
      logger.error(
          "Error explaining decision for transaction {}: {}", transactionId, e.getMessage());

      return ResponseEntity.status(500).body(Map.of("error", "Could not generate explanation"));
    }
  }
}
