package com.agenticfraud.engine.models;

import java.util.Map;

public record EnrichedTransactionWithML(
        EnrichedTransaction enriched,
        Double mlFraudScore,
        Map<String, Object> ragContext
) {}
