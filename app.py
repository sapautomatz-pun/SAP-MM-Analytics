# app.py
# SAP Automatz – Procurement Analytics (v26.4)
# Fix: Adaptive PDF table column sizing to avoid overlapping headers and content

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

# Try to import Pillow to measure image pixel dimensions
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# ---------- CONFIG ----------
LOGO_PATH = "sapautomatz_logo.png"
TAGLINE = "Automate. Analyze. Accelerate."
AI_TAGLINE = "Insights generated automatically by SAP Automatz AI Engine"
VALID_KEYS = {"SAPMM-00000000000000", "DEMO-ACCESS-12345"}
BRAND_BLUE = (26, 35, 126)  # #1a237e

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
    """Normalize and ASCII-safe string for PDF rendering."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("–", "-").replace("—", "-").replace("•", "-") \
         .replace("“", '"').replace("”", '"').replace("’", "'")
    s = s.replace("\t", " ").replace("\r", " ")
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

# ---------- IMAGE SIZING ----------
def get_image_height_mm(path, display_w_mm=170):
    fallback = 98.0
    if not PIL_AVAILABLE:
        return fallback
    try:
        with Image.open(path) as im:
            w_px, h_px = im.size
        if w_px == 0:
            return fallback
        return (h_px / w_px) * display_w_mm
    except Exception:
        return fallback

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
    cols_upper = {c.upper(): c for c in df_cols}
    for cand in candidates:
        if cand.upper() in cols_upper:
            return cols_upper[cand.upper()]
    for c in df_cols:
        cu = c.upper()
        for cand in candidates:
            if cand.upper() in cu:
                return c
    return None

def prepare_dataframe(df: pd.DataFrame):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    amt_col = detect_column(df.columns, ["AMOUNT", "AMT", "VALUE"])
    if amt_col and amt_col != "AMOUNT":
        df.rename(columns={amt_col: "AMOUNT"}, inplace=True)
    po_qty_col = detect_column(df.columns, ["PO_QTY", "POQTY", "QTY", "ORDER_QTY"])
    if po_qty_col and po_qty_col != "PO_QTY":
        df.rename(columns={po_qty_col: "PO_QTY"}, inplace=True)
    gr_qty_col = detect_column(df.columns, ["GR_QTY", "RECEIVED_QTY", "GRQTY", "RECV_QTY"])
    if gr_qty_col and gr_qty_col != "GR_QTY":
        df.rename(columns={gr_qty_col: "GR_QTY"}, inplace=True)
    po_date_col = detect_column(df.columns, ["PO_DATE", "ORDER_DATE", "DOC_DATE"])
    if po_date_col and po_date_col != "PO_DATE":
        df.rename(columns={po_date_col: "PO_DATE"}, inplace=True)
    gr_date_col = detect_column(df.columns, ["GR_DATE", "GOOD_RECEIVE_DATE", "GRN_DATE"])
    if gr_date_col and gr_date_col != "GR_DATE":
        df.rename(columns={gr_date_col: "GR_DATE"}, inplace=True)
    inv_date_col = detect_column(df.columns, ["INVOICE_DATE", "BILL_DATE", "INV_DATE"])
    if inv_date_col and inv_date_col != "INVOICE_DATE":
        df.rename(columns={inv_date_col: "INVOICE_DATE"}, inplace=True)
    gr_status_col = detect_column(df.columns, ["GR_STATUS", "GRSTATUS", "RECEIPT_STATUS", "RECV_STATUS"])
    if gr_status_col and gr_status_col != "GR_STATUS":
        df.rename(columns={gr_status_col: "GR_STATUS"}, inplace=True)

    # optional SLA columns might be present: SLA_DAYS, VENDOR_SLA, MATERIAL_SLA
    # normalize existence
    if "AMOUNT" not in df.columns:
        df["AMOUNT"] = 0.0
    if "CURRENCY" not in df.columns:
        df["CURRENCY"] = "INR"

    amounts = []
    currencies = []
    for _, r in df.iterrows():
        a, c = parse_amount_and_currency(r.get("AMOUNT", 0), r.get("CURRENCY", "INR"))
        amounts.append(a); currencies.append(c)
    df["AMOUNT_NUM"] = amounts
    df["CURRENCY_DETECTED"] = currencies

    if "PO_QTY" in df.columns:
        df["PO_QTY_NUM"] = df["PO_QTY"].apply(to_int_safe)
    else:
        df["PO_QTY_NUM"] = 0
    if "GR_QTY" in df.columns:
        df["GR_QTY_NUM"] = df["GR_QTY"].apply(to_int_safe)
    else:
        df["GR_QTY_NUM"] = 0

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

    if "GR_STATUS" in df.columns:
        df["GR_STATUS_NORM"] = df["GR_STATUS"].astype(str).str.lower().str.strip()
    else:
        df["GR_STATUS_NORM"] = np.where(df["GR_QTY_NUM"] >= df["PO_QTY_NUM"], "complete", "partial")

    # normalize SLA columns if present
    for sla_col in ["SLA_DAYS", "VENDOR_SLA", "MATERIAL_SLA"]:
        if sla_col in df.columns:
            df[sla_col] = pd.to_numeric(df[sla_col], errors="coerce")
            df.loc[df[sla_col] < 0, sla_col] = np.nan

    return df

# ---------- KPI / RISK / EFFICIENCY ----------
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

# ---------- SLA precedence helpers ----------
def build_vendor_sla_map_from_file(uploaded_file):
    """
    Expect CSV with columns: VENDOR,SLA_DAYS (case-insensitive).
    Returns dict vendor -> sla_days (int)
    """
    if uploaded_file is None:
        return {}
    try:
        vdf = pd.read_csv(uploaded_file)
        cols = {c.upper(): c for c in vdf.columns}
        vendor_col = None
        sla_col = None
        for k, v in cols.items():
            if k in ("VENDOR", "SUPPLIER", "VENDOR_NAME"):
                vendor_col = v
            if k in ("SLA_DAYS", "SLA", "ON_TIME_DAYS", "THRESHOLD"):
                sla_col = v
        if vendor_col is None or sla_col is None:
            return {}
        vdf[vendor_col] = vdf[vendor_col].astype(str).str.strip()
        vdf[sla_col] = pd.to_numeric(vdf[sla_col], errors="coerce")
        vdf = vdf.dropna(subset=[sla_col])
        mapping = dict(zip(vdf[vendor_col], vdf[sla_col].astype(int)))
        return mapping
    except Exception:
        return {}

def apply_threshold_precedence(df, vendor_sla_map=None, global_default=7):
    """
    Apply precedence to determine ON_TIME_THRESHOLD per row:
      1. SLA_DAYS (row)
      2. vendor_sla_map (uploaded)
      3. VENDOR_SLA (row)
      4. MATERIAL_SLA (row)
      5. global_default
    """
    df = df.copy()
    vendor_sla_map = vendor_sla_map or {}
    for c in ["SLA_DAYS", "VENDOR_SLA", "MATERIAL_SLA"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c] < 0, c] = np.nan

    def choose_threshold(r):
        if pd.notna(r.get("SLA_DAYS")):
            return int(r["SLA_DAYS"])
        v = r.get("VENDOR")
        if isinstance(v, str) and v in vendor_sla_map:
            try:
                return int(vendor_sla_map[v])
            except Exception:
                pass
        if pd.notna(r.get("VENDOR_SLA")):
            return int(r["VENDOR_SLA"])
        if pd.notna(r.get("MATERIAL_SLA")):
            return int(r["MATERIAL_SLA"])
        return int(global_default)

    df["ON_TIME_THRESHOLD"] = df.apply(choose_threshold, axis=1)
    return df

# ---------- VENDOR PERFORMANCE ----------
def compute_vendor_performance(df: pd.DataFrame):
    if "VENDOR" not in df.columns or df.empty:
        return pd.DataFrame()
    df_rows = df.copy()
    df_rows["gr_days"] = (df_rows["GR_DATE"] - df_rows["PO_DATE"]).dt.days
    df_rows["invoice_lag"] = (df_rows["INVOICE_DATE"] - df_rows["GR_DATE"]).dt.days
    df_rows["is_on_time"] = False
    mask = (~df_rows["GR_DATE"].isna()) & (~df_rows["PO_DATE"].isna())
    if mask.any():
        df_rows.loc[mask, "is_on_time"] = df_rows.loc[mask].apply(lambda r: (pd.notna(r["gr_days"]) and pd.notna(r["ON_TIME_THRESHOLD"]) and r["gr_days"] <= r["ON_TIME_THRESHOLD"]), axis=1)

    df_rows["is_partial"] = np.where(df_rows["GR_QTY_NUM"] < df_rows["PO_QTY_NUM"], 1, 0)

    g = df_rows.groupby("VENDOR").agg(
        total_po_qty=pd.NamedAgg(column="PO_QTY_NUM", aggfunc="sum"),
        total_gr_qty=pd.NamedAgg(column="GR_QTY_NUM", aggfunc="sum"),
        total_spend=pd.NamedAgg(column="AMOUNT_NUM", aggfunc="sum"),
        avg_gr_days=pd.NamedAgg(column="gr_days", aggfunc=lambda x: x.dropna().mean()),
        avg_invoice_lag=pd.NamedAgg(column="invoice_lag", aggfunc=lambda x: x.dropna().mean()),
        ontime_pct=pd.NamedAgg(column="is_on_time", aggfunc=lambda x: x.dropna().mean()),
        partial_pct=pd.NamedAgg(column="is_partial", aggfunc=lambda x: x.dropna().mean()),
        applied_sla_mean=pd.NamedAgg(column="ON_TIME_THRESHOLD", aggfunc=lambda x: int(np.round(np.nanmean(x))) if x.notna().any() else np.nan),
        rows_count=pd.NamedAgg(column="VENDOR", aggfunc="count")
    )
    g["fulfillment_rate"] = np.where(g["total_po_qty"] > 0, g["total_gr_qty"] / g["total_po_qty"], 0.0)
    g["ontime_pct"] = g["ontime_pct"].fillna(0.0) * 100.0
    g["partial_pct"] = g["partial_pct"].fillna(0.0) * 100.0
    g["avg_gr_days"] = g["avg_gr_days"].fillna(np.nan)
    g["avg_invoice_lag"] = g["avg_invoice_lag"].fillna(np.nan)
    g = g.reset_index().rename(columns={"index": "VENDOR"})
    out = g[["VENDOR", "applied_sla_mean", "total_po_qty", "total_gr_qty", "fulfillment_rate", "ontime_pct", "partial_pct", "avg_gr_days", "avg_invoice_lag", "total_spend", "rows_count"]]
    out = out.rename(columns={"applied_sla_mean": "SLA_days", "total_po_qty": "total_po_qty", "total_gr_qty": "total_gr_qty"})
    out = out.sort_values("total_spend", ascending=False).reset_index(drop=True)
    return out

# ---------- INSIGHTS & AI TEXT ----------
def currency_exposure_insight_extended(totals: dict):
    if not totals:
        return "Currency Exposure: No currency data available."
    total = sum(totals.values())
    if len(totals) == 1:
        cur = list(totals.keys())[0]
        amt = totals[cur]
        return f"Currency Exposure — Single currency ({cur}): {amt:,.0f} (100.0%)."
    items = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    parts = [f"{cur}: {amt:,.0f} ({amt/total*100:.1f}%)" for cur, amt in items[:4]]
    exposure_others = 100.0 - sum((amt/total*100.0) for _, amt in items[:1])
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
    ontime_best = vperf_df.sort_values("ontime_pct", ascending=False).iloc[0]
    avg_inv_lag = vperf_df["avg_invoice_lag"].dropna()
    avg_inv_lag_val = avg_inv_lag.mean() if not avg_inv_lag.empty else np.nan
    lines = [
        f"Vendor Performance: Best fulfillment: {best['VENDOR']} ({best['fulfillment_rate']*100:.1f}% filled).",
        f"Worst fulfillment: {worst['VENDOR']} ({worst['fulfillment_rate']*100:.1f}% filled).",
        f"Best on-time deliveries: {ontime_best['VENDOR']} ({ontime_best['ontime_pct']:.1f}% on-time).",
        f"Average invoice lag across vendors (days): {avg_inv_lag_val:.1f} (NaN if no invoice dates).",
        "Recommend engaging low-fulfillment vendors for corrective action and improving invoice processing for lagging suppliers."
    ]
    return " ".join(lines)

def generate_ai_text(k: dict):
    total = k.get("total_spend", 0.0)
    currency = k.get("dominant", "INR")
    top_v = list(k.get("top_v", {}).keys())[:3]
    top_v_text = ", ".join(top_v) if top_v else "no major vendors identified"
    totals = k.get("totals", {})
    if totals and len(totals) > 1:
        other = [c for c in totals.keys() if c != currency]
        exposure = sum(v for c, v in totals.items() if c != currency)
        exposure_pct = (exposure / (total + 1e-9)) * 100.0
        exposure_text = f" with {exposure_pct:.1f}% exposure in {', '.join(other[:3])}"
    else:
        exposure_text = ""
    risk_est = ""
    try:
        risk = compute_risk(k)
        risk_est = f" Risk Score: {risk['score']:.1f} ({risk['band']})."
    except Exception:
        risk_est = ""
    return (f"Total procurement spend was {total:,.2f} {currency}{exposure_text}. "
            f"Top vendors by spend: {top_v_text}.{risk_est} Overall procurement performance indicates opportunities in vendor optimization, delivery fulfillment, and invoice processing.")

# ---------- CHARTS ----------
def generate_dashboard_charts(k: dict, risk: dict, vendor_perf_df: pd.DataFrame):
    charts = []
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

# ---------- PDF helpers (improved adaptive draw_table) ----------
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

def _normalize_col_widths(pdf: PDF, col_w, ncols):
    usable = pdf.w - pdf.l_margin - pdf.r_margin  # mm
    if not col_w or len(col_w) != ncols:
        return [usable / ncols for _ in range(ncols)]
    total = sum(col_w)
    if total <= 0:
        return [usable / ncols for _ in range(ncols)]
    factor = usable / total
    w_scaled = [max(6.0, float(w) * factor) for w in col_w]
    s = sum(w_scaled)
    if abs(s - usable) > 0.01:
        diff = usable - s
        w_scaled[0] += diff
    return w_scaled

def draw_table(pdf: PDF, headers, rows, proposed_w=None, header_font_size=9, row_font_size=8, alignments=None):
    """
    Draw a table adaptively:
      - measure header+content widths (using get_string_width),
      - compute required widths, ensure Vendor column gets extra space if needed,
      - reduce numeric columns to minimum widths if necessary to fit usable width.
    """
    ncols = len(headers)
    usable = pdf.w - pdf.l_margin - pdf.r_margin  # mm
    # initial col widths (normalized)
    if proposed_w and len(proposed_w) == ncols:
        col_w = _normalize_col_widths(pdf, proposed_w, ncols)
    else:
        col_w = [usable / ncols for _ in range(ncols)]

    # prepare text samples per column: header + each row cell (converted to str)
    samples = []
    for i in range(ncols):
        col_texts = [sanitize_for_pdf(headers[i])]
        for r in rows:
            # ensure we can index
            if i < len(r):
                col_texts.append("" if r[i] is None else str(r[i]))
        samples.append(col_texts)

    # measure required width per column using the larger of header_font_size and row_font_size
    measure_font_size = max(header_font_size, row_font_size)
    pdf.set_font("Helvetica", size=measure_font_size)
    padding_mm = 6.0  # mm padding per column (left+right approx)
    required = []
    for col_texts in samples:
        max_w = 0.0
        for t in col_texts:
            # treat newline parts separately: take max linear width among parts
            parts = t.split("\n")
            for p in parts:
                w = pdf.get_string_width(p)
                if w > max_w:
                    max_w = w
        req = max_w + padding_mm
        required.append(req)

    # impose sensible minimums (mm)
    min_vendor = 40.0  # vendor column minimum
    min_numeric = 10.0  # numeric columns minimum
    # identify vendor column index heuristically: header contains "Vendor" (case-insensitive)
    vendor_idx = 0
    for i, h in enumerate(headers):
        if "VENDOR" in h.upper():
            vendor_idx = i
            break

    # ensure required vendor width >= min_vendor
    required[vendor_idx] = max(required[vendor_idx], min_vendor)
    # ensure numeric minima for other columns
    for i in range(ncols):
        if i != vendor_idx:
            required[i] = max(required[i], min_numeric)

    total_req = sum(required)
    # if fits, use required (but may want to preserve proposed pattern: we'll use required)
    if total_req <= usable:
        col_w = required.copy()
        # if leftover, give leftover to vendor column to make it roomy
        leftover = usable - sum(col_w)
        if leftover > 1.0:
            col_w[vendor_idx] += leftover
    else:
        # need to shrink columns to fit usable width
        # start with required widths, reduce non-vendor columns down to min_numeric first
        col_w = required.copy()
        reducible = sum(col_w) - usable
        # list of indices we can reduce (prefer numeric columns)
        reducible_indices = [i for i in range(ncols) if i != vendor_idx]
        # attempt to reduce numeric cols proportionally but not below min_numeric
        for i in reducible_indices:
            allow = col_w[i] - min_numeric
            take = min(allow, reducible * (col_w[i] / sum(col_w[j] for j in reducible_indices)))
            col_w[i] -= take
            reducible -= take
            if reducible <= 0:
                break
        # if still reducible left, reduce vendor down to its min (but keep readable)
        if reducible > 0:
            allow_v = col_w[vendor_idx] - min_vendor
            take_v = min(allow_v, reducible)
            col_w[vendor_idx] -= take_v
            reducible -= take_v
        # if still reducible, as last resort scale all columns proportionally to usable
        if reducible > 0:
            factor = usable / sum(col_w)
            col_w = [max(6.0, w * factor) for w in col_w]
            # tiny adjustment to sum exactly usable
            s = sum(col_w)
            if abs(s - usable) > 0.01:
                col_w[0] += (usable - s)

    # Draw header (single-line cells) using header font (prevents header wrap split)
    header_h = max(7, header_font_size * 0.6 + 4)
    pdf.set_font("Helvetica", "B", header_font_size)
    pdf.set_text_color(*BRAND_BLUE)
    for h, w in zip(headers, col_w):
        pdf.cell(w, header_h, sanitize_for_pdf(h), border=1, align="C")
    pdf.ln(header_h)
    pdf.set_text_color(0, 0, 0)

    # Prepare alignments default: vendor left, numbers right
    if alignments is None:
        alignments = ['L'] + ['R'] * (ncols - 1)

    # Draw rows: for each row compute lines needed and render with multi_cell, respecting col_w
    for row in rows:
        sanitized = [sanitize_for_pdf("" if v is None else str(v)) for v in row]
        # measure needed lines per cell (using row_font_size)
        pdf.set_font("Helvetica", size=row_font_size)
        est_lines = []
        for text, w in zip(sanitized, col_w):
            parts = text.split("\n")
            inner_w = max(1.0, w - 2.0)
            max_lines = 0
            for part in parts:
                if part.strip() == "":
                    needed = 1
                else:
                    text_width = pdf.get_string_width(part)
                    needed = int(np.ceil(text_width / inner_w))
                    needed = max(1, needed)
                max_lines = max(max_lines, needed)
            est_lines.append(max_lines)
        lines_needed = max(est_lines) if est_lines else 1
        line_h = max(4.0, row_font_size * 0.35 + 2)
        cell_h = lines_needed * line_h + 2
        ensure_pdf_space(pdf, cell_h + 2)
        # render each cell
        for i, (text, w) in enumerate(zip(sanitized, col_w)):
            align = alignments[i] if i < len(alignments) else 'L'
            x = pdf.get_x()
            y = pdf.get_y()
            pdf.multi_cell(w, line_h, text, border=1, align=align)
            pdf.set_xy(x + w, y)
        pdf.ln(cell_h)

# ---------- generate_pdf (uses draw_table) ----------
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

    # Procurement Insights as bullet points
    ensure_pdf_space(pdf, 40 + len(insights) * 6)
    pdf.set_text_color(*BRAND_BLUE); pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, sanitize_for_pdf("Procurement Insights (Expanded)"), ln=True)
    pdf.set_text_color(0, 0, 0); pdf.set_font("Helvetica", "", 11)
    for ins in insights:
        ensure_pdf_space(pdf, 10)
        pdf.cell(6)
        pdf.multi_cell(0, 6, "- " + sanitize_for_pdf(ins))
    pdf.ln(4)

    # Executive summary
    ensure_pdf_space(pdf, 40)
    pdf.set_text_color(*BRAND_BLUE); pdf.set_font("Helvetica", "B", 14); pdf.cell(0, 8, sanitize_for_pdf("Executive Summary"), ln=True)
    pdf.set_text_color(0, 0, 0); pdf.set_font("Helvetica", "", 12); pdf.multi_cell(0, 7, sanitize_for_pdf(ai_text))
    pdf.ln(6)

    # Key Performance Metrics
    metrics = [
        ("Total Spend", f"{k['total_spend']:,.2f} {k['dominant']}"),
        ("Records", f"{k['records']}"),
        ("Vendors (Top shown)", f"{len(k['top_v'])}"),
        ("Materials (Top shown)", f"{len(k['top_m'])}"),
        ("Risk Score", f"{risk['score']:.1f} ({risk['band']})")
    ]
    ensure_pdf_space(pdf, 12 + len(metrics) * 8)
    pdf.set_text_color(*BRAND_BLUE); pdf.set_font("Helvetica", "B", 14); pdf.cell(0, 8, sanitize_for_pdf("Key Performance Metrics"), ln=True)
    pdf.ln(2)
    col_w = [90, 90]
    pdf.set_font("Helvetica", "B", 10)
    draw_table(pdf, ["Metric", "Value"], [(m[0], m[1]) for m in metrics], proposed_w=col_w, header_font_size=10, row_font_size=10, alignments=['L','R'])
    pdf.ln(6)

    # Vendor Performance - include SLA_days column
    pdf.set_text_color(*BRAND_BLUE); pdf.set_font("Helvetica", "B", 14); pdf.cell(0, 8, sanitize_for_pdf("Vendor Performance (Top)"), ln=True)
    pdf.set_text_color(0, 0, 0); pdf.set_font("Helvetica", "", 9)
    if not vendor_perf_df.empty:
        vdf = vendor_perf_df.copy().head(12)
        headers = ["Vendor", "SLA (d)", "PO Qty", "GR Qty", "Fulfill %", "On-time %", "Partial %", "Avg GR d", "Avg Inv Lag (d)", "Total Spend"]
        proposed_w = [65, 12, 12, 12, 14, 14, 14, 12, 14, 25]
        rows = []
        for _, row in vdf.iterrows():
            vendor = row.get("VENDOR", "")
            sla = int(row.get("SLA_days")) if pd.notna(row.get("SLA_days")) else ""
            poq = int(row.get("total_po_qty") or 0)
            grq = int(row.get("total_gr_qty") or 0)
            fulfill = row.get("fulfillment_rate") * 100.0 if pd.notna(row.get("fulfillment_rate")) else np.nan
            ontime = row.get("ontime_pct") if pd.notna(row.get("ontime_pct")) else np.nan
            partial = row.get("partial_pct") if pd.notna(row.get("partial_pct")) else np.nan
            avg_gr = row.get("avg_gr_days") if pd.notna(row.get("avg_gr_days")) else ""
            invlag = row.get("avg_invoice_lag") if pd.notna(row.get("avg_invoice_lag")) else ""
            tspend = row.get("total_spend") if pd.notna(row.get("total_spend")) else 0.0
            cells = [
                vendor,
                f"{sla}" if sla != "" else "",
                f"{poq}",
                f"{grq}",
                f"{fulfill:.1f}" if not pd.isna(fulfill) else "",
                f"{ontime:.1f}" if not pd.isna(ontime) else "",
                f"{partial:.1f}" if not pd.isna(partial) else "",
                f"{avg_gr:.1f}" if isinstance(avg_gr, (int, float)) and not pd.isna(avg_gr) else "",
                f"{invlag:.1f}" if isinstance(invlag, (int, float)) and not pd.isna(invlag) else "",
                f"{tspend:,.2f}"
            ]
            rows.append(cells)
        draw_table(pdf, headers, rows, proposed_w=proposed_w, header_font_size=9, row_font_size=8,
                   alignments=['L','R','R','R','R','R','R','R','R','R'])
    else:
        pdf.multi_cell(0, 6, "Vendor performance data not available.")
    pdf.ln(6)

    # Material Category Performance
    pdf.set_text_color(*BRAND_BLUE); pdf.set_font("Helvetica", "B", 14); pdf.cell(0, 8, sanitize_for_pdf("Material Category Performance"), ln=True)
    pdf.set_text_color(0, 0, 0); pdf.set_font("Helvetica", "", 9)
    mat = k.get("top_m", {}) or {}
    if mat:
        items = list(mat.items())
        items = sorted(items, key=lambda x: x[1], reverse=True)[:12]
        headers = ["Material", "Spend"]
        proposed_w = [120, 70]
        rows = [[m, f"{val:,.2f}"] for m, val in items]
        draw_table(pdf, headers, rows, proposed_w=proposed_w, header_font_size=9, row_font_size=9, alignments=['L','R'])
    else:
        pdf.multi_cell(0, 6, "Material category data not available.")
    pdf.ln(6)

    # Charts
    if charts:
        pdf.add_page()
        pdf.set_text_color(*BRAND_BLUE); pdf.set_font("Helvetica", "B", 14); pdf.cell(0, 8, sanitize_for_pdf("Dashboard Charts"), ln=True)
        pdf.set_text_color(0, 0, 0); pdf.ln(4)
        captions = ["Monthly Spend Trend", "Top Vendors (Bar)", "Risk Breakdown", "Vendor Fulfillment Rate", "Material Spend Distribution"]
        display_w_mm = 170.0
        for idx, ch in enumerate(charts):
            img_h_mm = get_image_height_mm(ch, display_w_mm)
            total_needed = img_h_mm + 18
            ensure_pdf_space(pdf, total_needed)
            y = pdf.get_y()
            try:
                pdf.image(ch, x=20, y=y, w=display_w_mm)
            except Exception:
                try:
                    pdf.image(ch, x=20, w=display_w_mm)
                except Exception:
                    continue
            pdf.set_y(y + img_h_mm + 6)
            pdf.set_font("Helvetica", "I", 10)
            pdf.set_text_color(80, 80, 120)
            cap = captions[idx] if idx < len(captions) else f"Figure {idx+1}"
            pdf.cell(0, 6, sanitize_for_pdf(f"Figure {idx+1}: {cap}"), ln=True, align="C")
            pdf.set_text_color(0, 0, 0); pdf.set_font("Helvetica", "", 12)
            pdf.ln(6)

    # Risk Summary
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

# ---------- APP FLOW (same as before) ----------
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

st.sidebar.header("SLA / Thresholds")
default_on_time_days = st.sidebar.slider("Default on-time threshold (days)", min_value=0, max_value=90, value=7, help="Default PO -> GR SLA in days if none specified per PO/vendor/material")
st.sidebar.markdown("You can also upload a Vendor SLA CSV with columns `VENDOR,SLA_DAYS` (optional).")
vendor_sla_file = st.sidebar.file_uploader("Optional: Vendor SLA file (CSV)", type=["csv"], key="vendor_sla_upload")

file = st.file_uploader("Upload Procurement File (CSV/XLSX) — include SLA_DAYS / VENDOR_SLA / MATERIAL_SLA if you want per-row thresholds", type=["csv", "xlsx"])
if not file:
    st.info("Please upload your procurement extract.")
    st.stop()

company = st.text_input("Enter Company Name", "ABC Manufacturing Pvt Ltd")

try:
    df = pd.read_excel(file) if file.name.lower().endswith(".xlsx") else pd.read_csv(file)
    k = compute_kpis(df)

    # Build vendor SLA map from uploaded file (if present)
    vendor_sla_map = build_vendor_sla_map_from_file(vendor_sla_file)

    # Apply threshold precedence and update df inside k
    df_with_sla = apply_threshold_precedence(k["df"], vendor_sla_map=vendor_sla_map, global_default=default_on_time_days)
    k["df"] = df_with_sla

    # recompute risk / ai text based on updated df
    risk = compute_risk(k)
    ai_text = generate_ai_text(k)

    vendor_perf_df = compute_vendor_performance(k["df"])

    # Prepare insights
    insights = []
    insights.append(currency_exposure_insight_extended(k.get("totals", {})))
    insights.append(monthly_quarterly_trend_insight_extended(k.get("monthly", {})))
    insights.append(material_spend_insight_extended(k.get("top_m", {}), k.get("total_spend", 0.0)))
    insights.append(supplier_relationship_insight_extended(k.get("top_v", {}), k.get("total_spend", 0.0)))
    insights.append(vendor_performance_insight_extended(vendor_perf_df))
    eff_summary = compute_efficiency_summary(k["df"])
    insights.append(efficiency_insight_summary(eff_summary))
    insights.append(current_month_snapshot(k.get("monthly", {}), k.get("top_v", {})))

    # Display insights & summary
    st.markdown("## Procurement Insights (Expanded)")
    for ins in insights:
        st.write(ins)
    st.caption(AI_TAGLINE)

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

    # Vendor Performance - show SLA (applied) column
    st.markdown("### Vendor Performance")
    if not vendor_perf_df.empty:
        display_df = vendor_perf_df.copy()
        display_df = display_df.rename(columns={
            "SLA_days":"SLA (d)",
            "total_po_qty":"PO Qty",
            "total_gr_qty":"GR Qty",
            "fulfillment_rate":"Fulfillment",
            "ontime_pct":"On-time (%)",
            "partial_pct":"Partial (%)",
            "avg_gr_days":"Avg GR (d)",
            "avg_invoice_lag":"Avg Inv Lag (d)",
            "total_spend":"Total Spend",
            "rows_count":"Rows"
        })
        display_df["Fulfillment"] = (display_df["Fulfillment"] * 100.0).round(1)
        display_df["On-time (%)"] = display_df["On-time (%)"].round(1)
        display_df["Partial (%)"] = display_df["Partial (%)"].round(1)
        display_df["Avg GR (d)"] = display_df["Avg GR (d)"].round(1)
        display_df["Avg Inv Lag (d)"] = display_df["Avg Inv Lag (d)"].round(1)
        display_df["Total Spend"] = display_df["Total Spend"].round(2)
        st.dataframe(display_df[["VENDOR","SLA (d)","PO Qty","GR Qty","Fulfillment","On-time (%)","Partial (%)","Avg GR (d)","Avg Inv Lag (d)","Total Spend"]].sort_values("Total Spend", ascending=False))
    else:
        st.write("No vendor quantity/GR/invoice data available.")

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

    st.markdown("### Material Category Performance")
    mat_perf = (k.get("top_m", {}) or {})
    if mat_perf:
        st.dataframe(pd.DataFrame.from_dict(mat_perf, orient="index", columns=["Spend"]).sort_values("Spend", ascending=False).head(10))
    else:
        st.write("No material performance data available.")

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

    if st.button("📄 Generate PDF Report"):
        pdf_buf = generate_pdf(ai_text, insights, k, risk, charts, company, vendor_perf_df)
        safe_name = (company.strip().replace(" ", "_") or "Company")
        st.download_button("Download Report", pdf_buf, file_name=f"{safe_name}_Procurement_Report.pdf", mime="application/pdf")

except Exception:
    st.error("Error generating report:")
    st.text(traceback.format_exc())
