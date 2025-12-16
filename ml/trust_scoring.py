from collections import Counter

# -----------------------------
# Source reliability map
# -----------------------------
SOURCE_SCORE = {
    "gov": 1.0,
    "kaggle": 0.7,
    "other": 0.5
}

# -----------------------------
# Attribute trust calculation
# -----------------------------
def attribute_trust(values_with_sources, ml_confidence=0.0):
    """
    values_with_sources: list of tuples -> [(value, source), ...]
    ml_confidence: float (0 to 1)
    """

    if not values_with_sources:
        return 0.0

    # Completeness
    non_empty = [v for v, _ in values_with_sources if str(v).strip()]
    completeness_score = 1.0 if non_empty else 0.0

    # Source reliability (max source wins)
    source_scores = [
        SOURCE_SCORE.get(src, 0.5) for _, src in values_with_sources
    ]
    source_score = max(source_scores) if source_scores else 0.0

    # Agreement score
    values = [str(v).strip().lower() for v, _ in values_with_sources if v]
    count = Counter(values)

    if len(count) == 1 and values:
        agreement_score = 1.0
    elif len(count) > 1:
        agreement_score = 0.3
    else:
        agreement_score = 0.0

    # Weighted trust
    trust = (
        0.35 * source_score +
        0.35 * agreement_score +
        0.2  * completeness_score +
        0.1  * ml_confidence
    )

    return round(trust, 3)

# -----------------------------
# Overall record trust
# -----------------------------
def compute_trust(record_fields, ml_confidence=0.0):
    """
    record_fields = {
        "name": [(value, source), ...],
        "address": [(value, source), ...],
        "pincode": [(value, source), ...]
    }
    """

    trust_breakdown = {}

    for field, values in record_fields.items():
        trust_breakdown[field] = attribute_trust(
            values,
            ml_confidence=ml_confidence
        )

    overall_trust = round(
        sum(trust_breakdown.values()) / len(trust_breakdown), 3
    )

    return {
        "field_trust": trust_breakdown,
        "overall_trust": overall_trust
    }
