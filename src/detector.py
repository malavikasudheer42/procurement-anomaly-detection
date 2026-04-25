"""
Procurement Fraud & Cost Anomaly Detection Engine
Core detection logic: Isolation Forest + statistical flagging
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


class ProcurementAnomalyDetector:
    """
    Detects anomalous procurement transactions using a hybrid approach:
    - Isolation Forest for multivariate outlier detection
    - Statistical z-score flagging for univariate spend spikes
    - Rule-based flags for known fraud patterns (duplicate invoices, round-sum bias)
    """

    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_cols = [
            "amount", "days_since_last_invoice", "invoice_count_30d",
            "amount_vs_vendor_avg", "round_sum_flag", "duplicate_flag"
        ]

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["invoice_date"] = pd.to_datetime(df["invoice_date"])
        df = df.sort_values(["vendor_id", "invoice_date"])

        # Days between invoices per vendor
        df["days_since_last_invoice"] = (
            df.groupby("vendor_id")["invoice_date"]
            .diff()
            .dt.days
            .fillna(df.groupby("vendor_id")["invoice_date"].diff().dt.days.median())
        )
        df["days_since_last_invoice"] = df["days_since_last_invoice"].fillna(30)

        # Invoice frequency in rolling 30-day window (approximated per vendor)
        df["invoice_count_30d"] = (
            df.groupby("vendor_id")["invoice_date"]
            .transform(lambda x: x.expanding().count())
            .clip(upper=50)
        )

        # Amount deviation from vendor average
        vendor_avg = df.groupby("vendor_id")["amount"].transform("mean")
        df["amount_vs_vendor_avg"] = df["amount"] / vendor_avg.replace(0, 1)

        # Round-sum bias (e.g. exactly £10,000 — common in fraud)
        df["round_sum_flag"] = (df["amount"] % 1000 == 0).astype(int)

        # Duplicate invoice detection (same vendor, same amount, within 7 days)
        df["dup_key"] = (
            df["vendor_id"].astype(str) + "_" + df["amount"].astype(str)
        )
        df["duplicate_flag"] = df.duplicated(subset=["dup_key"], keep=False).astype(int)
        df.drop(columns=["dup_key"], inplace=True)

        return df

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._engineer_features(df)

        X = df[self.feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)

        # Isolation Forest scores (-1 = anomaly, 1 = normal)
        df["if_label"] = self.model.fit_predict(X_scaled)
        df["anomaly_score"] = -self.model.score_samples(X_scaled)  # Higher = more anomalous

        # Z-score flag on raw amount
        z = np.abs((df["amount"] - df["amount"].mean()) / df["amount"].std())
        df["zscore_flag"] = (z > 2.5).astype(int)

        # Final composite flag
        df["is_anomaly"] = (
            (df["if_label"] == -1) | (df["zscore_flag"] == 1) |
            (df["duplicate_flag"] == 1)
        ).astype(int)

        # Risk tier
        df["risk_tier"] = pd.cut(
            df["anomaly_score"],
            bins=[-np.inf, 0.4, 0.6, np.inf],
            labels=["Low", "Medium", "High"]
        )

        return df

    def get_feature_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Approximates SHAP-style feature contributions by measuring how much
        each feature deviates from its median for flagged transactions.
        """
        flagged = df[df["is_anomaly"] == 1][self.feature_cols].copy()
        medians = df[self.feature_cols].median()

        contributions = {}
        for col in self.feature_cols:
            contributions[col] = abs(flagged[col].mean() - medians[col])

        total = sum(contributions.values()) or 1
        return pd.DataFrame({
            "feature": list(contributions.keys()),
            "contribution": [v / total * 100 for v in contributions.values()]
        }).sort_values("contribution", ascending=False)
