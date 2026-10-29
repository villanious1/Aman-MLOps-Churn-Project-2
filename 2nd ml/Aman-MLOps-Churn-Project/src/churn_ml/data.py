"""Reproducible synthetic subscription data for offline experimentation."""
import numpy as np
import pandas as pd

NUMERIC = ["tenure_months", "monthly_fee", "support_tickets"]
CATEGORICAL = ["contract_type", "payment_method", "has_streaming"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "churned"


def make_customer_data(n_samples: int = 1500, random_state: int = 21) -> pd.DataFrame:
    if n_samples < 40:
        raise ValueError("n_samples must be at least 40")
    rng = np.random.default_rng(random_state)
    tenure = rng.integers(1, 73, n_samples)
    fee = rng.uniform(25, 125, n_samples).round(2)
    tickets = rng.poisson(1.4, n_samples).clip(0, 10)
    contract = rng.choice(["month_to_month", "one_year", "two_year"], n_samples,
                          p=[0.56, 0.27, 0.17])
    payment = rng.choice(["card", "bank_transfer", "digital_wallet"], n_samples)
    streaming = rng.choice([True, False], n_samples)
    logit = (
        1.15 - 0.045 * tenure + 0.018 * (fee - 70)
        + 0.48 * tickets
        + 1.05 * (contract == "month_to_month")
        - 0.45 * (contract == "two_year")
        + 0.25 * (payment == "digital_wallet")
        + rng.normal(0, 0.7, n_samples)
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)
    return pd.DataFrame({
        "tenure_months": tenure,
        "monthly_fee": fee,
        "support_tickets": tickets,
        "contract_type": contract,
        "payment_method": payment,
        "has_streaming": streaming,
        "churned": churn,
    })
