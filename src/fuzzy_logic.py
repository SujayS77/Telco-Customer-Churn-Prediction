
"""
Fuzzy logic module for Telco Customer Churn Prediction.

Converts the model's churn probability into overlapping fuzzy
risk categories and derives a transparent retention priority.
"""


def membership_low(value, start, end):
    """Membership decreases from 1 to 0 between start and end."""
    if value <= start:
        return 1.0
    if value >= end:
        return 0.0
    return (end - value) / (end - start)


def membership_high(value, start, end):
    """Membership increases from 0 to 1 between start and end."""
    if value <= start:
        return 0.0
    if value >= end:
        return 1.0
    return (value - start) / (end - start)


def membership_medium(value, left, peak, right):
    """Triangular fuzzy membership function."""
    if value <= left or value >= right:
        return 0.0
    if value == peak:
        return 1.0
    if value < peak:
        return (value - left) / (peak - left)
    return (right - value) / (right - peak)


def assess_fuzzy_risk(churn_probability, tenure):
    """
    Return fuzzy memberships and a rule-based retention priority.

    churn_probability: number between 0 and 1
    tenure: customer tenure in months, between 0 and 72
    """
    if not 0 <= churn_probability <= 1:
        raise ValueError("Churn probability must be between 0 and 1.")

    if not 0 <= tenure <= 72:
        raise ValueError("Tenure must be between 0 and 72 months.")

    # Fuzzy sets for churn probability
    risk = {
        "Low": membership_low(churn_probability, 0.25, 0.50),
        "Medium": membership_medium(churn_probability, 0.25, 0.50, 0.75),
        "High": membership_high(churn_probability, 0.50, 0.80),
    }

    # Fuzzy sets for customer tenure
    tenure_sets = {
        "New": membership_low(tenure, 6, 18),
        "Established": membership_medium(tenure, 6, 18, 36),
        "Loyal": membership_high(tenure, 18, 36),
    }

    # Fuzzy rules: each rule contributes a strength to a priority.
    # AND uses min; multiple rules for a category use max.
    priority = {"Low": 0.0, "Medium": 0.0, "High": 0.0}

    # Rule 1: High churn risk -> high retention priority.
    priority["High"] = risk["High"]

    # Rule 2: Medium risk + new customer -> high priority.
    priority["High"] = max(
        priority["High"],
        min(risk["Medium"], tenure_sets["New"]),
    )

    # Rule 3: Medium risk + established customer -> medium priority.
    priority["Medium"] = min(
        risk["Medium"], tenure_sets["Established"]
    )

    # Rule 4: Medium risk + loyal customer -> medium priority.
    priority["Medium"] = max(
        priority["Medium"],
        min(risk["Medium"], tenure_sets["Loyal"]),
    )

    # Rule 5: Low churn risk -> low priority.
    priority["Low"] = risk["Low"]

    # Select the strongest fuzzy priority.
    # This is a rule-based category, not an ML probability.
    priority_label = max(priority, key=priority.get)

    return {
        "risk_membership": risk,
        "tenure_membership": tenure_sets,
        "priority_strength": priority,
        "retention_priority": priority_label,
    }
