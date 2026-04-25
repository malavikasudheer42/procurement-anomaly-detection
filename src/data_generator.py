"""
Synthetic Procurement Data Generator
Generates realistic invoice/transaction data with seeded anomalies
for demonstration and testing purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


VENDORS = {
    "V001": ("Apex Construction Ltd", "Construction"),
    "V002": ("BuildRight Services", "Construction"),
    "V003": ("TechSupply Co", "IT Equipment"),
    "V004": ("DataSoft Solutions", "Software"),
    "V005": ("PrimeCatering Ltd", "Facilities"),
    "V006": ("SafeGuard Security", "Security"),
    "V007": ("GreenSpace Maintenance", "Grounds"),
    "V008": ("ElectroPro Systems", "Electrical"),
    "V009": ("CleanCo Services", "Cleaning"),
    "V010": ("OfficeFirst Ltd", "Stationery"),
}

CATEGORIES = [v[1] for v in VENDORS.values()]
DEPARTMENTS = ["Finance", "Operations", "IT", "Facilities", "Procurement", "Legal"]


def generate_transactions(
    n_normal: int = 500,
    n_anomalies: int = 30,
    seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    random.seed(seed)

    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = (end_date - start_date).days

    records = []

    # --- Normal transactions ---
    for i in range(n_normal):
        vendor_id = rng.choice(list(VENDORS.keys()))
        vendor_name, category = VENDORS[vendor_id]
        days_offset = int(rng.uniform(0, date_range))
        invoice_date = start_date + timedelta(days=days_offset)

        # Category-realistic amounts
        base_amounts = {
            "Construction": (5000, 80000),
            "IT Equipment": (1000, 25000),
            "Software": (500, 15000),
            "Facilities": (200, 5000),
            "Security": (1000, 20000),
            "Grounds": (300, 8000),
            "Electrical": (2000, 40000),
            "Cleaning": (150, 3000),
            "Stationery": (50, 2000),
        }
        lo, hi = base_amounts.get(category, (500, 50000))
        amount = round(float(rng.uniform(lo, hi)), 2)

        records.append({
            "invoice_id": f"INV-{10000 + i}",
            "invoice_date": invoice_date,
            "vendor_id": vendor_id,
            "vendor_name": vendor_name,
            "category": category,
            "department": random.choice(DEPARTMENTS),
            "amount": amount,
            "anomaly_type": "None",
        })

    # --- Seeded anomalies ---
    anomaly_types = [
        "duplicate_invoice",
        "amount_spike",
        "round_sum_bias",
        "high_frequency_billing",
        "new_vendor_large_payment",
    ]

    for j in range(n_anomalies):
        atype = anomaly_types[j % len(anomaly_types)]
        vendor_id = rng.choice(list(VENDORS.keys()))
        vendor_name, category = VENDORS[vendor_id]
        days_offset = int(rng.uniform(0, date_range))
        invoice_date = start_date + timedelta(days=days_offset)

        if atype == "duplicate_invoice":
            # Exact duplicate of a recent invoice
            ref = records[rng.integers(0, len(records))]
            amount = ref["amount"]
        elif atype == "amount_spike":
            # 10-20x normal
            lo, hi = 100000, 500000
            amount = round(float(rng.uniform(lo, hi)), 2)
        elif atype == "round_sum_bias":
            # Suspiciously round number
            amount = float(rng.choice([5000, 10000, 25000, 50000, 100000]))
        elif atype == "high_frequency_billing":
            # Normal amount but will cluster
            amount = round(float(rng.uniform(500, 5000)), 2)
            invoice_date = start_date + timedelta(days=int(rng.integers(1, 14)))
        elif atype == "new_vendor_large_payment":
            vendor_id = "V999"
            vendor_name = "UnknownSupplier Ltd"
            category = "Unknown"
            amount = round(float(rng.uniform(80000, 250000)), 2)

        records.append({
            "invoice_id": f"INV-A{j:04d}",
            "invoice_date": invoice_date,
            "vendor_id": vendor_id,
            "vendor_name": vendor_name,
            "category": category,
            "department": random.choice(DEPARTMENTS),
            "amount": amount,
            "anomaly_type": atype,
        })

    df = pd.DataFrame(records).sort_values("invoice_date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_transactions()
    df.to_csv("data/sample_transactions.csv", index=False)
    print(f"Generated {len(df)} transactions → data/sample_transactions.csv")
