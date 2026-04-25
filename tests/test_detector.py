"""
Unit tests for ProcurementAnomalyDetector
Run: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detector import ProcurementAnomalyDetector
from src.data_generator import generate_transactions


@pytest.fixture
def sample_df():
    return generate_transactions(n_normal=200, n_anomalies=15, seed=0)


@pytest.fixture
def detector():
    return ProcurementAnomalyDetector(contamination=0.05)


class TestDataGenerator:
    def test_generates_correct_row_count(self, sample_df):
        assert len(sample_df) == 215

    def test_required_columns_present(self, sample_df):
        required = {"invoice_id", "invoice_date", "vendor_id", "vendor_name",
                    "category", "department", "amount"}
        assert required.issubset(set(sample_df.columns))

    def test_no_negative_amounts(self, sample_df):
        assert (sample_df["amount"] >= 0).all()

    def test_invoice_ids_unique(self, sample_df):
        assert sample_df["invoice_id"].is_unique


class TestDetector:
    def test_output_has_expected_columns(self, detector, sample_df):
        result = detector.fit_predict(sample_df)
        for col in ["is_anomaly", "anomaly_score", "risk_tier", "duplicate_flag"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_is_anomaly_is_binary(self, detector, sample_df):
        result = detector.fit_predict(sample_df)
        assert set(result["is_anomaly"].unique()).issubset({0, 1})

    def test_anomaly_score_is_positive(self, detector, sample_df):
        result = detector.fit_predict(sample_df)
        assert (result["anomaly_score"] >= 0).all()

    def test_risk_tier_valid_values(self, detector, sample_df):
        result = detector.fit_predict(sample_df)
        valid = {"Low", "Medium", "High"}
        actual = set(result["risk_tier"].dropna().unique())
        assert actual.issubset(valid)

    def test_detects_at_least_some_anomalies(self, detector, sample_df):
        result = detector.fit_predict(sample_df)
        assert result["is_anomaly"].sum() > 0

    def test_feature_importance_sums_to_100(self, detector, sample_df):
        result = detector.fit_predict(sample_df)
        fi = detector.get_feature_importance(result)
        assert abs(fi["contribution"].sum() - 100.0) < 1.0

    def test_round_sum_flag_correct(self, detector, sample_df):
        result = detector.fit_predict(sample_df)
        round_sums = result[result["round_sum_flag"] == 1]["amount"]
        assert (round_sums % 1000 == 0).all()

    def test_contamination_affects_flag_count(self, sample_df):
        det_low = ProcurementAnomalyDetector(contamination=0.02)
        det_high = ProcurementAnomalyDetector(contamination=0.10)
        low = det_low.fit_predict(sample_df)["is_anomaly"].sum()
        high = det_high.fit_predict(sample_df)["is_anomaly"].sum()
        # Higher contamination should not produce fewer flags
        assert high >= low

    def test_handles_single_vendor_df(self, detector):
        """Edge case: all transactions from one vendor"""
        df = generate_transactions(n_normal=50, n_anomalies=5, seed=1)
        df["vendor_id"] = "V001"
        df["vendor_name"] = "Single Vendor Ltd"
        result = detector.fit_predict(df)
        assert "is_anomaly" in result.columns

    def test_reproducible_with_same_seed(self):
        df1 = generate_transactions(seed=99)
        df2 = generate_transactions(seed=99)
        pd.testing.assert_frame_equal(df1, df2)
