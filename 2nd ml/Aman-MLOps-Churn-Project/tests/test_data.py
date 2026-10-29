import pytest
from src.churn_ml.data import FEATURES, TARGET, make_customer_data


def test_generated_data_has_expected_schema():
    df = make_customer_data(100, 8)
    assert list(df.columns) == FEATURES + [TARGET]
    assert len(df) == 100
    assert set(df[TARGET].unique()).issubset({0, 1})


def test_generation_is_reproducible():
    assert make_customer_data(60, 3).equals(make_customer_data(60, 3))


def test_small_dataset_is_rejected():
    with pytest.raises(ValueError):
        make_customer_data(10)
