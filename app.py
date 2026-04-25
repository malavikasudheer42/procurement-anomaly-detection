"""
Procurement Fraud & Cost Anomaly Detection Engine
Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detector import ProcurementAnomalyDetector
from src.data_generator import generate_transactions

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Procurement Anomaly Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.main { background-color: #0a0e1a; }

.metric-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 12px;
}

.metric-card.red { border-left: 4px solid #ef4444; }
.metric-card.amber { border-left: 4px solid #f59e0b; }
.metric-card.green { border-left: 4px solid #10b981; }
.metric-card.blue { border-left: 4px solid #3b82f6; }

.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #f9fafb;
    margin: 0;
}

.metric-label {
    font-size: 0.78rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

.risk-high { color: #ef4444; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
.risk-medium { color: #f59e0b; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
.risk-low { color: #10b981; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }

.section-header {
    font-size: 0.72rem;
    color: #4b5563;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid #1f2937;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

.stDataFrame { font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; }

.flag-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
}

div[data-testid="stMetric"] label { color: #6b7280 !important; font-size: 0.75rem !important; }
div[data-testid="stMetric"] div { color: #f9fafb !important; font-family: 'IBM Plex Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans", color="#9ca3af"),
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    contamination = st.slider(
        "Model sensitivity (contamination rate)",
        min_value=0.01, max_value=0.15, value=0.05, step=0.01,
        help="Proportion of transactions expected to be anomalous"
    )
    st.markdown("---")
    st.markdown("**Data source**")
    use_upload = st.checkbox("Upload your own CSV", value=False)
    uploaded_file = None
    if use_upload:
        uploaded_file = st.file_uploader(
            "Upload transactions CSV",
            type=["csv"],
            help="Required columns: invoice_id, invoice_date, vendor_id, vendor_name, category, department, amount"
        )
    st.markdown("---")
    st.markdown("**Filters**")
    filter_anomalies_only = st.checkbox("Show flagged transactions only", value=False)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#4b5563; line-height:1.6'>
    <b style='color:#6b7280'>Model</b><br>
    Isolation Forest (n=200)<br>
    + Z-score (σ > 2.5)<br>
    + Rule-based flags<br><br>
    <b style='color:#6b7280'>Built by</b><br>
    Malavika Sudheer<br>
    github.com/malavika-sudheer
    </div>
    """, unsafe_allow_html=True)

# ── Load & run ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_and_detect(contamination: float, file_bytes=None):
    if file_bytes is not None:
        import io
        df_raw = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df_raw = generate_transactions(n_normal=500, n_anomalies=30)

    detector = ProcurementAnomalyDetector(contamination=contamination)
    df_result = detector.fit_predict(df_raw)
    fi = detector.get_feature_importance(df_result)
    return df_result, fi

file_bytes = uploaded_file.read() if uploaded_file else None
df, feature_importance = load_and_detect(contamination, file_bytes)

if filter_anomalies_only:
    display_df = df[df["is_anomaly"] == 1].copy()
else:
    display_df = df.copy()

# ── KPIs ──────────────────────────────────────────────────────────────────────
n_total = len(df)
n_flagged = df["is_anomaly"].sum()
flagged_value = df[df["is_anomaly"] == 1]["amount"].sum()
total_spend = df["amount"].sum()
flag_rate = n_flagged / n_total * 100
vendor_concentration = df.groupby("vendor_id")["amount"].sum().max() / total_spend * 100

st.markdown(
    "<h2 style='font-family:IBM Plex Mono;color:#f9fafb;font-size:1.3rem;margin-bottom:4px'>"
    "PROCUREMENT ANOMALY ENGINE</h2>"
    "<p style='color:#4b5563;font-size:0.8rem;margin-top:0'>Isolation Forest · Statistical Scoring · Rule-Based Flagging</p>",
    unsafe_allow_html=True
)
st.markdown("<div class='section-header'>PORTFOLIO OVERVIEW</div>", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="metric-card blue">
      <p class="metric-value">{n_total:,}</p>
      <p class="metric-label">Total Transactions</p>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card red">
      <p class="metric-value">{n_flagged:,}</p>
      <p class="metric-label">Flagged Anomalies</p>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card amber">
      <p class="metric-value">£{flagged_value:,.0f}</p>
      <p class="metric-label">At-Risk Spend</p>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card green">
      <p class="metric-value">£{total_spend:,.0f}</p>
      <p class="metric-label">Total Spend Analysed</p>
    </div>""", unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="metric-card amber">
      <p class="metric-value">{vendor_concentration:.1f}%</p>
      <p class="metric-label">Top Vendor Concentration</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts Row 1 ─────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>SPEND ANALYSIS</div>", unsafe_allow_html=True)
col_left, col_right = st.columns([2, 1])

with col_left:
    # Timeline scatter
    fig_timeline = px.scatter(
        df.sort_values("invoice_date"),
        x="invoice_date",
        y="amount",
        color="is_anomaly",
        color_discrete_map={0: "#1f2937", 1: "#ef4444"},
        size="amount",
        size_max=18,
        hover_data=["invoice_id", "vendor_name", "category", "anomaly_score"],
        labels={"is_anomaly": "Anomaly", "amount": "Amount (£)", "invoice_date": "Date"},
        title="Transaction Timeline — Flagged vs Normal"
    )
    fig_timeline.update_layout(**PLOTLY_THEME, height=320, showlegend=False)
    fig_timeline.update_traces(marker=dict(opacity=0.75, line=dict(width=0)))
    st.plotly_chart(fig_timeline, use_container_width=True)

with col_right:
    # Vendor spend bar
    vendor_spend = (
        df.groupby("vendor_name")["amount"]
        .sum()
        .sort_values(ascending=True)
        .tail(8)
    )
    fig_vendor = go.Figure(go.Bar(
        x=vendor_spend.values,
        y=vendor_spend.index,
        orientation="h",
        marker_color="#3b82f6",
        marker_line_width=0,
    ))
    fig_vendor.update_layout(**PLOTLY_THEME, height=320,
                             title="Vendor Spend Concentration",
                             xaxis_title="Total Spend (£)", yaxis_title="")
    st.plotly_chart(fig_vendor, use_container_width=True)

# ── Charts Row 2 ─────────────────────────────────────────────────────────────
col_a, col_b, col_c = st.columns(3)

with col_a:
    # Anomaly score distribution
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=df[df["is_anomaly"] == 0]["anomaly_score"],
        name="Normal", marker_color="#1f2937", nbinsx=30, opacity=0.9
    ))
    fig_dist.add_trace(go.Histogram(
        x=df[df["is_anomaly"] == 1]["anomaly_score"],
        name="Anomaly", marker_color="#ef4444", nbinsx=30, opacity=0.9
    ))
    fig_dist.update_layout(**PLOTLY_THEME, height=260, barmode="overlay",
                           title="Anomaly Score Distribution", showlegend=True)
    st.plotly_chart(fig_dist, use_container_width=True)

with col_b:
    # Category breakdown
    cat_flags = df.groupby("category").agg(
        total=("amount", "sum"), flagged=("is_anomaly", "sum")
    ).reset_index()
    cat_flags["flag_rate"] = cat_flags["flagged"] / cat_flags["total"].replace(0, 1) * 100

    fig_cat = px.bar(
        cat_flags.sort_values("flagged", ascending=False),
        x="category", y="flagged",
        color="flag_rate",
        color_continuous_scale=["#1f2937", "#f59e0b", "#ef4444"],
        title="Flagged Count by Category",
        labels={"flagged": "Flags", "category": "", "flag_rate": "Flag Rate (%)"}
    )
    fig_cat.update_layout(**PLOTLY_THEME, height=260, coloraxis_showscale=False)
    st.plotly_chart(fig_cat, use_container_width=True)

with col_c:
    # Feature importance
    fig_fi = px.bar(
        feature_importance.head(6),
        x="contribution", y="feature",
        orientation="h",
        color="contribution",
        color_continuous_scale=["#1f2937", "#3b82f6", "#ef4444"],
        title="Detection Signal Strength",
        labels={"contribution": "% Contribution", "feature": ""}
    )
    fig_fi.update_layout(**PLOTLY_THEME, height=260, coloraxis_showscale=False)
    st.plotly_chart(fig_fi, use_container_width=True)

# ── Flagged Transactions Table ────────────────────────────────────────────────
st.markdown("<div class='section-header'>FLAGGED TRANSACTION LEDGER</div>", unsafe_allow_html=True)

table_cols = [
    "invoice_id", "invoice_date", "vendor_name", "category",
    "department", "amount", "anomaly_score", "risk_tier",
    "duplicate_flag", "round_sum_flag", "anomaly_type"
]

table_df = display_df[table_cols].copy()
table_df["invoice_date"] = table_df["invoice_date"].dt.strftime("%Y-%m-%d")
table_df["amount"] = table_df["amount"].apply(lambda x: f"£{x:,.2f}")
table_df["anomaly_score"] = table_df["anomaly_score"].round(4)
table_df = table_df.rename(columns={
    "invoice_id": "Invoice ID",
    "invoice_date": "Date",
    "vendor_name": "Vendor",
    "category": "Category",
    "department": "Dept",
    "amount": "Amount",
    "anomaly_score": "Risk Score",
    "risk_tier": "Risk Tier",
    "duplicate_flag": "Dup",
    "round_sum_flag": "Round Sum",
    "anomaly_type": "Seeded Type",
})

flagged_table = table_df[display_df["is_anomaly"] == 1] if not filter_anomalies_only else table_df

st.dataframe(
    flagged_table.sort_values("Risk Score", ascending=False),
    use_container_width=True,
    height=400,
)

# ── Download ──────────────────────────────────────────────────────────────────
csv_export = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇ Export full results as CSV",
    data=csv_export,
    file_name="anomaly_detection_results.csv",
    mime="text/csv",
)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='color:#374151;font-size:0.72rem;font-family:IBM Plex Mono'>",
    unsafe_allow_html=True
)
