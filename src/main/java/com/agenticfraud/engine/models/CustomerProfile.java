package com.agenticfraud.engine.models;

import java.math.BigDecimal;
import java.util.List;

public record CustomerProfile(
    String customerId,
    BigDecimal averageTransactionAmount,
    BigDecimal dailySpendingLimit,
    List<String> transactionCategories,
    String primaryLocation,
    String riskLevel // LOW, MEDIUM, HIGH
    ) { }
