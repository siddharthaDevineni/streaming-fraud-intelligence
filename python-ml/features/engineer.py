import polars as pl
from config import settings
from models.schemas import EnrichedTransaction
import structlog

logger = structlog.get_logger()


def extract_features(enriched: EnrichedTransaction) -> dict:
    txn = enriched.transaction
    profile = enriched.customerProfile
    velocity = enriched.velocityCount or 1

    avg_amount = profile.averageTransactionAmount if profile else 100.0
    daily_limit = profile.dailySpendingLimit if profile else 1000.0
    risk_level = profile.riskLevel if profile else "UNKNOWN"
    primary_location = profile.primaryLocation if profile else ""
    typical_categories = profile.transactionCategories if profile else []

    amount_ratio = txn.amount / avg_amount if avg_amount > 0 else 1.0
    daily_limit_ratio = txn.amount / daily_limit if daily_limit > 0 else 0.0
    is_amount_unusual = 1 if txn.amount > (avg_amount * 3) else 0

    is_high_velocity = 1 if velocity >= settings.high_velocity_threshold else 0
    velocity_squared = velocity ** 2

    is_unknown_location = 1 if "unknown" in txn.location.lower() else 0
    is_primary_location = 1 if txn.location.lower() == primary_location.lower() else 0
    is_different_city = 1 if (not is_primary_location and not is_unknown_location) else 0

    is_online = 1 if txn.merchantCategory == "ONLINE" else 0
    is_typical_category = 1 if txn.merchantCategory in typical_categories else 0
    is_suspicious_merchant = 1 if "suspicious" in txn.merchantId.lower() else 0

    hour = txn.timestamp.hour
    is_off_hours = 1 if (hour < 6 or hour >= 22) else 0
    is_weekend = 1 if txn.timestamp.weekday() >= 5 else 0

    is_high_risk_customer = 1 if risk_level == "HIGH" else 0
    is_low_risk_customer = 1 if risk_level == "LOW" else 0

    metadata = txn.metadata or {}
    is_bot_device = 1 if "bot" in str(metadata.get("deviceId", "")).lower() else 0
    is_rapid_fire = 1 if metadata.get("rapidFire", False) else 0

    return {
        "amount_ratio": round(amount_ratio, 4),
        "daily_limit_ratio": round(daily_limit_ratio, 4),
        "is_amount_unusual": is_amount_unusual,
        "velocity_count": velocity,
        "velocity_squared": velocity_squared,
        "is_high_velocity": is_high_velocity,
        "is_unknown_location": is_unknown_location,
        "is_primary_location": is_primary_location,
        "is_different_city": is_different_city,
        "is_online": is_online,
        "is_typical_category": is_typical_category,
        "is_suspicious_merchant": is_suspicious_merchant,
        "hour": hour,
        "is_off_hours": is_off_hours,
        "is_weekend": is_weekend,
        "is_high_risk_customer": is_high_risk_customer,
        "is_low_risk_customer": is_low_risk_customer,
        "is_bot_device": is_bot_device,
        "is_rapid_fire": is_rapid_fire,
    }


def to_polars_frame(features: dict) -> pl.DataFrame:
    return pl.DataFrame([features])


FEATURE_COLUMNS = [
    "amount_ratio", "daily_limit_ratio", "is_amount_unusual",
    "velocity_count", "velocity_squared", "is_high_velocity",
    "is_unknown_location", "is_primary_location", "is_different_city",
    "is_online", "is_typical_category", "is_suspicious_merchant",
    "hour", "is_off_hours", "is_weekend",
    "is_high_risk_customer", "is_low_risk_customer",
    "is_bot_device", "is_rapid_fire",
]