# app.py
# SAP Automatz – Procurement Analytics v20
# Fix: restore missing insight functions and tighten up insight assembly

import os
import io
import re
import datetime
import tempfile
import traceback
import unicodedata
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from fpdf import FPDF

# ---------- CONFIG ----------
LOGO_PATH = "sapautomatz_logo.png"
TAGLINE = "Automate. Analyze. Accelerate."
AI_TAGLINE = "Insights generated automatically by SAP Automatz AI Engine"
VALID_KEYS = {"SAPMM-00000000000000", "DEMO-ACCESS-12345"}
BRAND_BLUE = (26, 35, 126)  # #1a237e
ON_TIME_WINDOW_DAYS = 7  # threshold to consider GR on-time (adjustable)

# ---------- STREAMLIT SETUP ----------
st.set_page_config(page_title="SAP Automatz – Procurement Analytics", layout="wide")
st.markdown("<style>.stApp header{visibility:hidden}</style>", unsafe_allow_html=True)
col_logo, col_title = st.columns([1, 3])
with col_logo:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=110)
with col_title:
    st.markdown(
        f"<h2 style='color:#1a237e;margin-bottom:0'>SAP Automatz – Procurement Analytics</h2>"
        f"<p style='margin-top:0;color:#555;'>{TAGLINE}</p>",
        unsafe_allow_html=True,
    )
st.divider()
st.session_state.setdefault("verified", False)

# ---------- UTIL ----------
def sanitize_for_pdf(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("–", "-").replace("—", "-").replace("•", "-").replace("“", '"').replace("”", '"').replace("’", "'")
    s_norm = unicodedata.normalize("NFKD", s)
    s_ascii = s_norm.encode("ascii", "ignore").decode("ascii")
    s_ascii = re.sub(r"\s+", " ", s_ascii).strip()
    return s_ascii

def to_float_safe(x):
    try:
        if pd.isna(x):
            return 0.0
        s = str(x).replace(",", "")
        return float(re.sub(r"[^\d.\-]", "", s)) if s not in ("", ".", "-", "-.") else 0.0
    except Exception:
        return 0.0

def to_int_safe(x):
    try:
        if pd.isna(x):
            return 0
        s = str(x).replace(",", "")
        return int(float(re.sub(r"[^\d.\-]", "", s))) if s not in ("", ".", "-", "-.") else 0
    except Exception:
        return 0

def parse_date_safe(x):
    try:
        return pd.to_datetime(x, errors="coerce")
    except Exception:
        return pd.NaT

# ---------- DATA HELPERS ----------
def parse_amount_and_currency(value, fallback="INR"):
    sym_map = {"₹": "INR", "Rs": "INR", "$": "USD", "€": "EUR", "£": "GBP"}
    if pd.isna(value):
        return 0.0, fallback
    s = str(value)
    detected = fallback
    for sym, code in sym_map.items():
        if sym in s:
            detected = code
            s = s.replace(sym, "")
    s = re.sub(r"[^\d.\-]", "", s)
    try:
        return float(s) if s not in ("", ".", "-", "-.") else 0.0, detected
    except Exception:
        return 0.0, detected

def detect_column(df_cols, candidates):
    """Return first matching column name in df_cols for a list of possible names (case-insensitive)."""
    cols_upper = {c.upper(): c for c in df_cols}
    for cand in candidates:
        if cand.upper() in cols_upper:
            return cols_upper[cand.upper()]
    # fuzzy: find containing keywords
    for c in df_cols:
        cu = c.upper()
        for cand in candidates:
            if cand.upper() in cu:
                return c
    return None

def prepare_dataframe(df: pd.DataFrame):
    df = df.copy()
    # normalize column names
    df.columns = [c.strip() for c in df.columns]

    # detect key columns (allow flexible names)
    amt_col = detect_column(df.columns, ["AMOUNT", "AMT", "VALUE"])
    if amt_col and amt_col != "AMOUNT":
        df.rename(columns={amt_col: "AMOUNT"}, inplace=True)
    # quantity columns
    po_qty_col = detect_column(df.columns, ["PO_QTY", "POQTY", "QTY", "ORDER_QTY", "PO_QTY"])
    if po_qty_col and po_qty_col != "PO_QTY":
        df.rename(columns={po_qty_col: "PO_QTY"}, inplace=True)
    gr_qty_col = detect_column(df.columns, ["GR_QTY", "RECEIVED_QTY", "GRQTY", "RECV_QTY"])
    if gr_qty_col and gr_qty_col != "GR_QTY":
        df.rename(columns={gr_qty_col: "GR_QTY"}, inplace=True)
    # dates
    po_date_col = detect_column(df.columns, ["PO_DATE", "ORDER_DATE", "DOC_DATE"])
    if po_date_col and po_date_col != "PO_DATE":
        df.rename(columns={po_date_col: "PO_DATE"}, inplace=True)
    gr_date_col = detect_column(df.columns, ["GR_DATE", "GOOD_RECEIVE_DATE", "GRN_DATE"])
    if gr_date_col and gr_date_col != "GR_DATE":
        df.rename(columns={gr_date_col: "GR_DATE"}, inplace=True)
    inv_date_col = detect_column(df.columns, ["INVOICE_DATE", "BILL_DATE", "INV_DATE"])
    if inv_date_col and inv_date_col != "INVOICE_DATE":
        df.rename(columns={inv_date_col: "INVOICE_DATE"}, inplace=True)
    # status
    gr_status_col = detect_column(df.columns, ["GR_STATUS", "GRSTATUS", "RECEIPT_STATUS", "RECV_STATUS"])
    if gr_status_col and gr_status_col != "GR_STATUS":
        df.rename(columns={gr_status_col: "GR_STATUS"}, inplace=True)

    # ensure columns exist
    if "AMOUNT" not in df.columns:
        df["AMOUNT"] = 0.0
    if "CURRENCY" not in df.columns:
        df["CURRENCY"] = "INR"

    # numeric amount + detected currency
    amounts = []
    currencies = []
    for _, r in df.iterrows():
        a, c = parse_amount_and_currency(r.get("AMOUNT", 0), r.get("CURRENCY", "INR"))
        amounts.append(a); currencies.append(c)
    df["AMOUNT_NUM"] = amounts
    df["CURRENCY_DETECTED"] = currencies

    # quantities
    if "PO_QTY" in df.columns:
        df["PO_QTY_NUM"] = df["PO_QTY"].apply(to_int_safe)
    else:
        df["PO_QTY_NUM"] = 0
    if "GR_QTY" in df.columns:
        df["GR_QTY_NUM"] = df["GR_QTY"].apply(to_int_safe)
    else:
        df["GR_QTY_NUM"] = 0

    # dates
    if "PO_DATE" in df.columns:
        df["PO_DATE"] = pd.to_datetime(df["PO_DATE"], errors="coerce")
    else:
        df["PO_DATE"] = pd.NaT
    if "GR_DATE" in df.columns:
        df["GR_DATE"] = pd.to_datetime(df["GR_DATE"], errors="coerce")
    else:
        df["GR_DATE"] = pd.NaT
    if "INVOICE_DATE" in df.columns:
        df["INVOICE_DATE"] = pd.to_datetime(df["INVOICE_DATE"], errors="coerce")
    else:
        df["INVOICE_DATE"] = pd.NaT

    # GR status normalization
    if "GR_STATUS" in df.columns:
        df["GR_STATUS_NORM"] = df["GR_STATUS"].astype(str).str.lower().str.strip()
    else:
        df["GR_STATUS_NORM"] = np.where(df["GR_QTY_NUM"] >= df["PO_QTY_NUM"], "complete", "partial")
    return df

# ---------- KPIs ----------
def compute_kpis(df: pd.DataFrame):
    df = prepare_dataframe(df)
    totals = df.groupby("CURRENCY_DETECTED")["AMOUNT_NUM"].sum().to_dict() if "CURRENCY_DETECTED" in df.columns else {}
    total_spend = float(sum(totals.values())) if totals else float(df["AMOUNT_NUM"].sum()) if "AMOUNT_NUM" in df.columns else 0.0
    dominant = max(totals, key=totals.get) if totals else "INR"
    top_v = df.groupby("VENDOR")["AMOUNT_NUM"].sum().nlargest(10).to_dict() if "VENDOR" in df.columns else {}
    top_m = df.groupby("MATERIAL")["AMOUNT_NUM"].sum().nlargest(10).to_dict() if "MATERIAL" in df.columns else {}
    monthly = {}
    if "PO_DATE" in df.columns:
        temp = df.dropna(subset=["PO_DATE"])
        if not temp.empty:
            temp["MONTH"] = temp["PO_DATE"].dt.to_period("M").astype(str)
            monthly = temp.groupby("MONTH")["AMOUNT_NUM"].sum().to_dict()
    return {"totals": totals, "total_spend": total_spend, "dominant": dominant,
            "top_v": top_v, "top_m": top_m, "monthly": monthly, "df": df, "records": len(df)}

# ---------- RISK & EFFICIENCY ----------
def compute_risk(k: dict):
    df = k.get("df", pd.DataFrame())
    total = float(k.get("total_spend", 0.0) or 0.0)
    if total == 0 or df.empty:
        return {"score": 0.0, "band": "Low", "breakdown": {}}
    dom = k.get("dominant", "INR")
    totals = k.get("totals", {})
    v = df.groupby("VENDOR")["AMOUNT_NUM"].sum() if "VENDOR" in df.columns else pd.Series(dtype=float)
    nv = len(v) if not v.empty else 0
    top_share = float(v.max()) / total if not v.empty else 1.0
    v_conc = (1 - top_share) * 100.0
    v_div = min(100.0, (nv / 50.0) * 100.0) if nv > 0 else 0.0
    c_expo = (totals.get(dom, 0.0) / total) * 100.0 if total else 100.0
    mvals = list(k.get("monthly", {}).values())
    m_vol = 100.0 * (1 - np.std(mvals) / (np.mean(mvals) + 1e-9)) if len(mvals) > 2 and np.mean(mvals) != 0 else 80.0
    score = float(np.clip((v_conc + v_div + c_expo + m_vol) / 4.0, 0.0, 100.0))
    band = "Low" if score >= 67 else ("Medium" if score >= 34 else "High")
    return {"score": score, "band": band, "breakdown": {
        "Vendor Concentration": v_conc, "Vendor Diversity": v_div,
        "Currency Exposure": c_expo, "Monthly Volatility": m_vol
    }}

def compute_efficiency_summary(df: pd.DataFrame):
    if "VENDOR" not in df.columns or df.empty:
        return {}
    avg_cost = df.groupby("VENDOR")["AMOUNT_NUM"].mean()
    total_spend = df.groupby("VENDOR")["AMOUNT_NUM"].sum()
    eff = {}
    for v in avg_cost.index:
        ac = float(avg_cost.loc[v])
        ts = float(total_spend.loc[v])
        units = ts / ac if ac != 0 else 0.0
        eff[v] = {"avg_cost": ac, "total_spend": ts, "units_est": units}
    return eff

# ---------- VENDOR PERFORMANCE ----------
def compute_vendor_performance(df: pd.DataFrame):
    if "VENDOR" not in df.columns or df.empty:
        return pd.DataFrame()
    g = df.groupby("VENDOR").agg(
        total_po_qty=pd.NamedAgg(column="PO_QTY_NUM", aggfunc="sum"),
        total_gr_qty=pd.NamedAgg(column="GR_QTY_NUM", aggfunc="sum"),
        total_spend=pd.NamedAgg(column="AMOUNT_NUM", aggfunc="sum"),
    )
    g["fulfillment_rate"] = np.where(g["total_po_qty"] > 0, g["total_gr_qty"] / g["total_po_qty"], np.nan)
    df_rows = df[["VENDOR", "PO_DATE", "GR_DATE", "INVOICE_DATE", "GR_STATUS_NORM", "PO_QTY_NUM", "GR_QTY_NUM", "AMOUNT_NUM"]].copy()
    df_rows["on_time"] = False
    mask = (~df_rows["GR_DATE"].isna()) & (~df_rows["PO_DATE"].isna())
    df_rows.loc[mask, "on_time"] = (df_rows.loc[mask, "GR_DATE"] - df_rows.loc[mask, "PO_DATE"]).dt.days <= ON_TIME_WINDOW_DAYS
    ontime = df_rows.groupby("VENDOR")["on_time"].mean().fillna(0) * 100.0
    g["on_time_pct"] = ontime
    df_rows["invoice_lag"] = np.nan
    mask2 = (~df_rows["INVOICE_DATE"].isna()) & (~df_rows["GR_DATE"].isna())
    df_rows.loc[mask2, "invoice_lag"] = (df_rows.loc[mask2, "INVOICE_DATE"] - df_rows.loc[mask2, "GR_DATE"]).dt.days
    invlag = df_rows.groupby("VENDOR")["invoice_lag"].mean().fillna(np.nan)
    g["avg_invoice_lag"] = invlag
    status_counts = df_rows.groupby(["VENDOR", "GR_STATUS_NORM"]).size().unstack(fill_value=0)
    for col in status_counts.columns:
        g[f"status_{col}"] = status_counts[col]
    for col in ["status_complete", "status_partial", "status_nan"]:
        if col not in g.columns:
            g[col] = 0
    g = g.sort_values("total_spend", ascending=False)
    g = g.reset_index().rename(columns={"index": "VENDOR"})
    return g

# ---------- INSIGHTS (complete) ----------
def currency_exposure_insight_extended(totals: dict):
    if not totals:
        return "Currency Exposure: No currency data available."
    total = sum(totals.values())
    items = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    parts = [f"{cur}: {amt:,.0f} ({amt/total*100:.1f}%)" for cur, amt in items[:4]]
    exposure_others = 100.0 - (items[0][1] / (total + 1e-9) * 100.0)
    risk_note = "High exposure to multiple currencies increases FX risk; consider hedging or invoicing strategies." if exposure_others > 20 else "Currency exposure appears concentrated; monitor FX rates."
    return "Currency Exposure — Multi-currency spend distribution with risk assessment: " + "; ".join(parts) + ". " + risk_note

def monthly_quarterly_trend_insight_extended(monthly: dict):
    if not monthly:
        return "Monthly/Quarterly Spend Trends: Not enough time-series data."
    months = sorted(monthly.keys())
    last_6 = months[-6:] if len(months) >= 6 else months
    vals = [monthly[m] for m in last_6]
    slope = np.polyfit(np.arange(len(vals)), vals, 1)[0] if len(vals) >= 2 else 0.0
    trend = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
    seasonal_note = "Seasonal fluctuations observed; spending peaks align with specific months or quarters." if len(monthly) >= 12 else ""
    return f"Monthly/Quarterly Spend Trends: Recent trend is {trend}. {seasonal_note}"

def material_spend_insight_extended(mat_perf: dict, total: float):
    if not mat_perf:
        return "Material Spend: No material category data available."
    top = list(mat_perf.items())[:5]
    parts = [f"{m} ({v/total*100:.1f}%)" for m, v in top[:3]]
    dominance_note = "Top categories dominate spend; consider category sourcing strategies." if sum(v for _, v in top[:3]) / (total + 1e-9) > 0.5 else "Material spend is relatively diversified."
    return f"Material Spend: Top categories include {', '.join(parts)}. {dominance_note}"

def supplier_relationship_insight_extended(top_v: dict, total: float):
    if not top_v:
        return "Supplier Relationship Management: No vendor data available."
    top3 = list(top_v.items())[:3]
    pct = sum(v for _, v in top3) / (total + 1e-9) * 100.0
    return (f"Supplier Relationship Management: Top 3 vendors contribute {pct:.1f}% of total spend. "
            "High concentration indicates negotiation leverage but also supplier dependency risk. "
            "Recommend strategic supplier segmentation, performance SLAs, and contingency sourcing.")

def efficiency_insight_summary(eff_summary: dict):
    if not eff_summary:
        return "Efficiency: Insufficient data."
    sorted_v = sorted(eff_summary.items(), key=lambda x: x[1]["avg_cost"])
    best, worst = sorted_v[0], sorted_v[-1]
    units = worst[1]["units_est"]
    gap = worst[1]["avg_cost"] - best[1]["avg_cost"]
    est_savings = units * gap if units > 0 and gap > 0 else 0.0
    return (f"Efficiency: Most Efficient: {best[0]} at {best[1]['avg_cost']:.2f} cost/unit. "
            f"Least Efficient: {worst[0]} at {worst[1]['avg_cost']:.2f} cost/unit; "
            f"estimated annual savings if optimized: {est_savings:,.0f} (vendor currency).")

def current_month_snapshot(monthly, top_v):
    if not monthly:
        return "Current Month Snapshot: No monthly data available."
    months = sorted(monthly.keys())
    last = months[-1]
    last_val = monthly[last]
    if len(months) >= 2:
        prev_val = monthly[months[-2]]
        pct_change = ((last_val - prev_val) / (prev_val + 1e-9)) * 100
        trend = "increased" if pct_change > 0 else "decreased" if pct_change < 0 else "remained stable"
        top_vendor = list(top_v.keys())[0] if top_v else "N/A"
        return (f"Current Month Snapshot: Spend in {last} was {last_val:,.0f}, "
                f"which {trend} by {abs(pct_change):.1f}% from the prior month. Top vendor for the period was {top_vendor}.")
    else:
        return f"Current Month Snapshot: Spend in {last} was {last_val:,.0f} (insufficient prior data for comparison)."

def vendor_performance_insight_extended(vperf_df: pd.DataFrame):
    if vperf_df.empty:
        return "Vendor Performance: No vendor quantity/GR/invoice data available."
    valid = vperf_df.dropna(subset=["fulfillment_rate"])
    if valid.empty:
        return "Vendor Performance: Not enough PO/GR quantity data to compute fulfillment rates."
    best = valid.sort_values("fulfillment_rate", ascending=False).iloc[0]
    worst = valid.sort_values("fulfillment_rate", ascending=True).iloc[0]
    ontime_best = vperf_df.sort_values("on_time_pct", ascending=False).iloc[0]
    avg_inv_lag = vperf_df["avg_invoice_lag"].dropna()
    avg_inv_lag_val = avg_inv_lag.mean() if not avg_inv_lag.empty else np.nan
    lines = [
        f"Vendor Performance: Best fulfillment: {best['VENDOR']} ({best['fulfillment_rate']*100:.1f}% filled).",
        f"Worst fulfillment: {worst['VENDOR']} ({worst['fulfillment_rate']*100:.1f}% filled).",
        f"Best on-time deliveries: {ontime_best['VENDOR']} ({ontime_best['on_time_pct']:.1f}% on-time).",
        f"Average invoice lag across vendors (days): {avg_inv_lag_val:.1f} (NaN if no invoice dates).",
        "Recommend engaging low-fulfillment vendors for corrective action and improving invoice processing for lagging suppliers."
    ]
    return " ".join(lines)

# ---------- CHARTS ----------
def generate_dashboard_charts(k: dict, risk: dict, vendor_perf_df: pd.DataFrame):
    charts = []
    # Monthly trend
    try:
        if k.get("monthly"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                plt.figure(figsize=(8, 3.6))
                months, vals = list(k["monthly"].keys()), list(k["monthly"].values())
                plt.plot(months, vals, marker="o", linewidth=1.5)
                plt.title("Monthly Spend Trend")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout(); plt.savefig(tmp.name, bbox_inches="tight"); charts.append(tmp.name); plt.close()
    except Exception:
        plt.close()
    # Top vendors by spend
    try:
        if k.get("top_v"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                top5 = dict(list(k["top_v"].items())[:5])
                plt.figure(figsize=(8, 3.6))
                plt.bar(range(len(top5)), list(top5.values()), tick_label=list(top5.keys()))
                plt.title("Top 5 Vendors by Spend")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout(); plt.savefig(tmp.name, bbox_inches="tight"); charts.append(tmp.name); plt.close()
    except Exception:
        plt.close()
    # Risk breakdown
    try:
        if risk.get("breakdown"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                keys, vals = zip(*list(risk["breakdown"].items()))
                plt.figure(figsize=(8, 3.6))
                plt.bar(range(len(vals)), vals, tick_label=keys)
                plt.title("Risk Breakdown")
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout(); plt.savefig(tmp.name, bbox_inches="tight"); charts.append(tmp.name); plt.close()
    except Exception:
        plt.close()
    # Vendor fulfillment chart
    try:
        if not vendor_perf_df.empty:
            dfv = vendor_perf_df.copy().head(8)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                labels = dfv["VENDOR"].tolist()
                vals = (dfv["fulfillment_rate"].fillna(0).astype(float) * 100).tolist()
                plt.figure(figsize=(8, 3.6))
                plt.bar(range(len(labels)), vals)
                plt.title("Vendor Fulfillment Rate (%)")
                plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
                plt.tight_layout(); plt.savefig(tmp.name, bbox_inches="tight"); charts.append(tmp.name); plt.close()
    except Exception:
        plt.close()
    # Material pie
    try:
        mat = k.get("top_m", {})
        if mat:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                items = list(mat.items())[:10]
                labels = [i[0] for i in items]
                sizes = [i[1] for i in items]
                plt.figure(figsize=(6.5, 4))
                plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
                plt.title("Material Spend Distribution (Top 10)")
                plt.tight_layout(); plt.savefig(tmp.name, bbox_inches="tight"); charts.append(tmp.name); plt.close()
    except Exception:
        plt.close()
    return charts

# ---------- PDF helpers ----------
def ensure_pdf_space(pdf_obj: FPDF, needed_height_mm: float):
    try:
        bottom_limit = pdf_obj.h - pdf_obj.b_margin
    except Exception:
        bottom_limit = 280
    if pdf_obj.get_y() + needed_height_mm > bottom_limit:
        pdf_obj.add_page()

class PDF(FPDF):
    def header(self):
        if os.path.exists(LOGO_PATH):
            self.image(LOGO_PATH, 10, 8, 20)
        self.set_xy(32, 10)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*BRAND_BLUE)
        self.cell(0, 6, "SAP Automatz", ln=True)
        self.set_xy(32, 16)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(80, 80, 80)
        self.cell(0, 5, TAGLINE, ln=True)
        self.ln(8)
        self.line(10, 25, 200, 25)
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, f"SAP Automatz | Page {self.page_no()}", 0, 0, "C")

def generate_pdf(ai_text, insights, k, risk, charts, company, vendor_perf_df):
    pdf = PDF(); pdf.alias_nb_pages(); pdf.add_page()
    pdf.set_font("Helvetica", "B", 20); pdf.set_text_color(*BRAND_BLUE)
    pdf.cell(0, 15, sanitize_for_pdf("Procurement Analytics Report"), ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 12); pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, sanitize_for_pdf(f"Company: {company}"), ln=True, align="C")
    pdf.cell(0, 8, sanitize_for_pdf(f"Generated On: {datetime.date.today():%d-%b-%Y}"), ln=True, align="C")
    pdf.ln(6); pdf.set_font("Helvetica", "I", 9); pdf.cell(0, 6, sanitize_for_pdf(AI_TAGLINE), ln=True)
    pdf.ln(8)

    ensure_pdf_space(pdf, 40 + len(insights) * 7)
    pdf.set_text_color(*BRAND_BLUE); pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, sanitize_for_pdf("Procurement Insights (Expanded)"), ln=True)
    pdf.set_text_color(0, 0, 0); pdf.set_font("Helvetica", "", 12)
    for ins in insights:
        ensure_pdf_space(pdf, 10)
        pdf.multi_cell(0, 7, sanitize_for_pdf(ins))
    pdf.ln(6)

    ensure_pdf_space(pdf, 40)
    pdf.set_text_color(*BRAND_BLUE); pdf.set_font("Helvetica", "B", 14); pdf.cell(0, 8, sanitize_for_pdf("Executive Summary"), ln=True)
    pdf.set_text_color(0, 0, 0); pdf.set_font("Helvetica", "", 12); pdf.multi_cell(0, 7, sanitize_for_pdf(ai_text))
    pdf.ln(6)

    metrics = [
        f"Total Spend: {k['total_spend']:,.2f} {k['dominant']}",
        f"Records: {k['records']}",
        f"Vendors (Top shown): {len(k['top_v'])}",
        f"Materials (Top shown): {len(k['top_m'])}",
        f"Risk Score: {risk['score']:.1f} ({risk['band']})"
    ]
    ensure_pdf_space(pdf, 12 + len(metrics)*7)
    pdf.set_text_color(*BRAND_BLUE); pdf.set_font("Helvetica", "B", 14); pdf.cell(0, 8, sanitize_for_pdf("Key Performance Metrics"), ln=True)
    pdf.set_text_color(0, 0, 0); pdf.set_font("Helvetica", "", 12)
    for m in metrics:
        ensure_pdf_space(pdf, 8)
        pdf.cell(0, 7, "- " + sanitize_for_pdf(m), ln=True)
    pdf.ln(6)

    ensure_pdf_space(pdf, 40)
    pdf.set_text_color(*BRAND_BLUE); pdf.set_font("Helvetica", "B", 14); pdf.cell(0, 8, sanitize_for_pdf("Vendor Performance Summary"), ln=True)
    pdf.set_text_color(0, 0, 0); pdf.set_font("Helvetica", "", 12)
    if not vendor_perf_df.empty:
        valid = vendor_perf_df.dropna(subset=["fulfillment_rate"])
        if not valid.empty:
            top = valid.sort_values("fulfillment_rate", ascending=False).iloc[0]
            bot = valid.sort_values("fulfillment_rate", ascending=True).iloc[0]
            pdf.multi_cell(0, 7, sanitize_for_pdf(f"Top performing vendor: {top['VENDOR']} (Fulfillment {top['fulfillment_rate']*100:.1f}%; On-time {top['on_time_pct']:.1f}%)."))
            pdf.multi_cell(0, 7, sanitize_for_pdf(f"Lowest performing vendor: {bot['VENDOR']} (Fulfillment {bot['fulfillment_rate']*100:.1f}%; On-time {bot['on_time_pct']:.1f}%)."))
        else:
            pdf.multi_cell(0, 7, sanitize_for_pdf("Not enough quantity data to compute vendor fulfillment rates."))
    else:
        pdf.multi_cell(0, 7, sanitize_for_pdf("Vendor performance metrics not available."))
    pdf.ln(6)

    # Charts start on new page
    if charts:
        pdf.add_page()
        pdf.set_text_color(*BRAND_BLUE); pdf.set_font("Helvetica", "B", 14); pdf.cell(0, 8, sanitize_for_pdf("Dashboard Charts"), ln=True)
        pdf.set_text_color(0, 0, 0); pdf.ln(4)
        captions = ["Monthly Spend Trend", "Top Vendors (Bar)", "Risk Breakdown", "Vendor Fulfillment Rate", "Material Spend Distribution"]
        for idx, ch in enumerate(charts):
            ensure_pdf_space(pdf, 110)
            try:
                y = pdf.get_y() + 6
                pdf.image(ch, x=20, y=y, w=170)
                pdf.ln(98)
                pdf.set_font("Helvetica", "I", 10)
                pdf.set_text_color(80, 80, 120)
                cap = captions[idx] if idx < len(captions) else f"Figure {idx+1}"
                pdf.cell(0, 6, sanitize_for_pdf(f"Figure {idx+1}: {cap}"), ln=True, align="C")
                pdf.set_text_color(0, 0, 0); pdf.set_font("Helvetica", "", 12)
                pdf.ln(6)
            except Exception:
                continue

    ensure_pdf_space(pdf, 40)
    pdf.set_text_color(*BRAND_BLUE); pdf.set_font("Helvetica", "B", 14); pdf.cell(0, 8, sanitize_for_pdf("Risk Summary"), ln=True)
    pdf.set_text_color(0, 0, 0); pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 7, sanitize_for_pdf(f"Risk Score: {risk['score']:.1f} ({risk['band']})"), ln=True)
    for k_, v_ in risk.get("breakdown", {}).items():
        ensure_pdf_space(pdf, 8)
        pdf.cell(0, 7, "- " + sanitize_for_pdf(f"{k_}: {v_:.1f}"), ln=True)
    pdf.ln(8)

    ensure_pdf_space(pdf, 10)
    pdf.set_font("Helvetica", "I", 9); pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, sanitize_for_pdf(AI_TAGLINE), ln=True)

    out = io.BytesIO(pdf.output(dest="S").encode("latin-1", "ignore"))
    out.seek(0)
    return out

# ---------- APP FLOW ----------
st.subheader("🔐 Verify Access Key")
col1, col2 = st.columns([3, 1])
with col1:
    key = st.text_input("Enter access key", type="password")
with col2:
    if st.button("Verify"):
        if key and key.strip() in VALID_KEYS:
            st.session_state["verified"] = True
            st.success("Access verified.")
            st.rerun()
        else:
            st.error("Invalid key.")
if not st.session_state["verified"]:
    st.stop()

file = st.file_uploader("Upload Procurement File (CSV/XLSX)", type=["csv", "xlsx"])
if not file:
    st.info("Please upload your procurement extract.")
    st.stop()

company = st.text_input("Enter Company Name", "ABC Manufacturing Pvt Ltd")
try:
    df = pd.read_excel(file) if file.name.lower().endswith(".xlsx") else pd.read_csv(file)
    k = compute_kpis(df)
    risk = compute_risk(k)
    ai_text = generate_ai_text(k)

    # vendor perf DF
    vendor_perf_df = compute_vendor_performance(k["df"])

    # Build insights using defined functions
    insights = []
    insights.append(currency_exposure_insight_extended(k.get("totals", {})))
    insights.append(monthly_quarterly_trend_insight_extended(k.get("monthly", {})))
    insights.append(material_spend_insight_extended(k.get("top_m", {}), k.get("total_spend", 0.0)))
    insights.append(supplier_relationship_insight_extended(k.get("top_v", {}), k.get("total_spend", 0.0)))
    insights.append(vendor_performance_insight_extended(vendor_perf_df))
    eff_summary = compute_efficiency_summary(k["df"])
    insights.append(efficiency_insight_summary(eff_summary))
    insights.append(current_month_snapshot(k.get("monthly", {}), k.get("top_v", {})))

    # On-screen insights
    st.markdown("## Procurement Insights (Expanded)")
    for ins in insights:
        st.write(ins)
    st.caption(AI_TAGLINE)

    # Executive summary & metrics
    st.markdown("## Executive Summary")
    st.write(ai_text)
    st.markdown("### Key Performance Metrics")
    cols = st.columns(5)
    metrics_vals = [
        f"{k['total_spend']:,.2f} {k['dominant']}",
        k["records"],
        len(k["top_v"]),
        len(k["top_m"]),
        f"{risk['score']:.1f} ({risk['band']})"
    ]
    labels = ["Total Spend", "Records", "Vendors", "Materials", "Risk Score"]
    for c, label, val in zip(cols, labels, metrics_vals):
        c.metric(label, str(val))

    # Vendor performance on-screen table
    st.markdown("### Vendor Performance")
    if not vendor_perf_df.empty:
        display_df = vendor_perf_df.copy()
        display_df["fulfillment_rate_pct"] = (display_df["fulfillment_rate"] * 100).round(1)
        display_df["avg_invoice_lag"] = display_df["avg_invoice_lag"].round(1)
        st.dataframe(display_df[["VENDOR", "total_po_qty", "total_gr_qty", "fulfillment_rate_pct", "on_time_pct", "avg_invoice_lag", "total_spend"]].rename(
            columns={"total_po_qty":"PO Qty","total_gr_qty":"GR Qty","fulfillment_rate_pct":"Fulfillment (%)","on_time_pct":"On-time (%)","avg_invoice_lag":"Avg Inv Lag (days)","total_spend":"Total Spend"}
        ).sort_values("Total Spend", ascending=False))
    else:
        st.write("No vendor quantity/GR/invoice data available.")

    # Efficiency on-screen
    st.markdown("### Efficiency Analysis")
    if eff_summary:
        sorted_eff = sorted(eff_summary.items(), key=lambda x: x[1]["avg_cost"])
        best, worst = sorted_eff[0], sorted_eff[-1]
        units = worst[1]["units_est"]
        gap = worst[1]["avg_cost"] - best[1]["avg_cost"]
        est_savings = units * gap if units > 0 and gap > 0 else 0.0
        st.write(f"**Most Efficient:** {best[0]} — {best[1]['avg_cost']:.2f} cost/unit; total spend {best[1]['total_spend']:,.0f}.")
        st.write(f"**Least Efficient:** {worst[0]} — {worst[1]['avg_cost']:.2f} cost/unit; estimated annual savings ≈ {est_savings:,.0f} (vendor currency).")
    else:
        st.write("Efficiency data not available.")

    # Material performance on-screen
    st.markdown("### Material Category Performance")
    mat_perf = (k.get("top_m", {}) or {})
    if mat_perf:
        st.dataframe(pd.DataFrame.from_dict(mat_perf, orient="index", columns=["Spend"]).sort_values("Spend", ascending=False).head(10))
    else:
        st.write("No material performance data available.")

    # Charts
    st.markdown("### Dashboard Charts")
    charts = generate_dashboard_charts(k, risk, vendor_perf_df)
    if charts:
        captions = []
        if k.get("monthly"): captions.append("Monthly Spend Trend")
        if k.get("top_v"): captions.append("Top Vendors")
        if risk.get("breakdown"): captions.append("Risk Breakdown")
        if not vendor_perf_df.empty: captions.append("Vendor Fulfillment Rate")
        if k.get("top_m"): captions.append("Material Spend Distribution")
        st.image(charts, caption=captions, use_container_width=True)
    else:
        st.info("Not enough data to generate charts.")
    st.caption(AI_TAGLINE)

    # PDF generation
    if st.button("📄 Generate PDF Report"):
        pdf_buf = generate_pdf(ai_text, insights, k, risk, charts, company, vendor_perf_df)
        safe_name = (company.strip().replace(" ", "_") or "Company")
        st.download_button("Download Report", pdf_buf, file_name=f"{safe_name}_Procurement_Report.pdf", mime="application/pdf")

except Exception:
    st.error("Error generating report:")
    st.text(traceback.format_exc())
