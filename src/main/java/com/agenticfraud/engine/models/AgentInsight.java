package com.agenticfraud.engine.models;

import java.time.LocalDateTime;

public record AgentInsight(
    String agentType,
    String agentName,
    String analysis,
    double riskScore,
    double confidence,
    String reasoning,
    String recommendation,
    LocalDateTime timestamp) { }
