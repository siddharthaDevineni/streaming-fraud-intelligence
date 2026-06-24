package com.agenticfraud.engine.models;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.time.Instant;
import java.util.Map;

@JsonIgnoreProperties(ignoreUnknown = true)
public record MLPrediction(
        String transactionId,
        String customerId,
        double mlFraudScore,
        double lstmSequenceScore,
        double combinedScore,
        String modelVersion,
        Map<String, Object> shapExplanation,
        Map<String, Object> featuresUsed,
        Map<String, Object> ragContext,
        int inferenceLatencyMs,
        Instant timestamp
) {}
