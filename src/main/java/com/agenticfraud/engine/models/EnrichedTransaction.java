package com.agenticfraud.engine.models;

public record EnrichedTransaction(
    Transaction transaction, CustomerProfile customerProfile, Long velocityCount) {}