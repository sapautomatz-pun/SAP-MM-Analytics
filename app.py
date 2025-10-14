# app.py
# SAP Automatz – Procurement Analytics v18
# Fix: prevents PDF section overlap by ensuring page space before sections/images

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

# ---------- UTIL: sanitize text for PDF ----------
def sanitize_for_pdf(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("–", "-").replace("—", "-").replace("•", "-").replace("“", '"').replace("”", '"').replace("’", "'")
    s_norm = unicodedata.normalize("NFKD", s)
    s_ascii = s_norm.encode("ascii", "ignore").decode("ascii")
    s_ascii = re.sub(r"\s+", " ", s_ascii).strip()
    return s_ascii

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

def prepare_dataframe(df: pd.DataFrame):
    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]
    if "AMOUNT" not in df.columns:
        for c in df.columns:
            if "AMT" in c.upper():
                df.rename(columns={c: "AMOUNT"}, inplace=True)
                break
    if "CURRENCY" not in df.columns:
        df["CURRENCY"] = "INR"
    amounts = []
    currencies = []
    for _, r in df.iterrows():
        a, c = parse_amount_and_currency(r.get("AMOUNT", 0), r.get("CURRENCY", "INR"))
        amounts.append(a); currencies.append(c)
    df["AMOUNT_NUM"] = amounts
    df["CURRENCY_DETECTED"] = currencies
    return df

def compute_kpis(df: pd.DataFrame):
    df = prepare_dataframe(df)
    totals = df.groupby("CURRENCY_DETECTED")["AMOUNT_NUM"].sum().to_dict() if "CURRENCY_DETECTED" in df.columns else {}
    total_spend = float(sum(totals.values())) if totals else float(df["AMOUNT_NUM"].sum()) if "AMOUNT_NUM" in df.columns else 0.0
    dominant = max(totals, key=totals.get) if totals else "INR"
    top_v = df.groupby("VENDOR")["AMOUNT_NUM"].sum().nlargest(10).to_dict() if "VENDOR" in df.columns else {}
    top_m = df.groupby("MATERIAL")["AMOUNT_NUM"].sum().nlargest(10).to_dict() if "MATERIAL" in df.columns else {}
    monthly = {}
    if "PO_DATE" in df.columns:
        df["PO_DATE"] = pd.to_datetime(df["PO_DATE"], errors="coerce")
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

def compute_material_performance(df: pd.DataFrame):
    if "MATERIAL" not in df.columns or df.empty:
        return {}
    return df.groupby("MATERIAL")["AMOUNT_NUM"].sum().sort_values(ascending=False).to_dict()

# ---------- EXECUTIVE SUMMARY ----------
def generate_ai_text(k: dict):
    total = k.get("total_spend", 0.0)
    currency = k.get("dominant", "INR")
    top_v = list(k.get("top_v", {}).keys())[:3]
    top_v_text = ", ".join(top_v) if top_v else "no major vendors identified"
    totals = k.get("totals", {})
    if len(totals) > 1:
        other = [c for c in totals.keys() if c != currency]
        exposure = sum(v for c, v in totals.items() if c != currency)
        exposure_pct = (exposure / (total + 1e-9)) * 100.0
        exposure_text = f" with {exposure_pct:.1f}% exposure in {', '.join(other[:3])}"
    else:
        exposure_text = ""
    return (f"Total procurement spend was {total:,.2f} {currency}{exposure_text}. "
            f"Top vendors by spend: {top_v_text}. "
            f"Overall procurement performance indicates opportunities in vendor optimization and currency risk management.")

# ---------- EXTENDED INSIGHTS (Option B) ----------
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
        return (f"Current Month Snapshot: Spend in {last} was {last_val:,.0f}, which {trend} by {abs(pct_change):.1f}% from the prior month. Top vendor: {top_vendor}.")
    else:
        return f"Current Month Snapshot: Spend in {last} was {last_val:,.0f} (insufficient prior data)."

# ---------- PDF Class ----------
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

# ---------- CHARTS ----------
def generate_dashboard_charts(k: dict, risk: dict):
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

# ---------- PDF generate helper: ensure space ----------
def ensure_pdf_space(pdf_obj: FPDF, needed_height_mm: float):
    """Ensure there is needed_height_mm space left on the current page; if not, add a new page."""
    try:
        bottom_limit = pdf_obj.h - pdf_obj.b_margin
    except Exception:
        # fallback conservative limit
        bottom_limit = 280
    if pdf_obj.get_y() + needed_height_mm > bottom_limit:
        pdf_obj.add_page()

# ---------- PDF Generator ----------
def generate_pdf(ai_text, insights, k, risk, charts, company):
    pdf = PDF(); pdf.alias_nb_pages(); pdf.add_page()

    # Header block
    pdf.set_font("Helvetica", "B", 20); pdf.set_text_color(*BRAND_BLUE)
    pdf.cell(0, 15, sanitize_for_pdf("Procurement Analytics Report"), ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 12); pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, sanitize_for_pdf(f"Company: {company}"), ln=True, align="C")
    pdf.cell(0, 8, sanitize_for_pdf(f"Generated On: {datetime.date.today():%d-%b-%Y}"), ln=True, align="C")
    pdf.ln(6); pdf.set_font("Helvetica", "I", 9); pdf.cell(0, 6, sanitize_for_pdf(AI_TAGLINE), ln=True)
    pdf.ln(8)

    # Insights - ensure space for block of text (approx height)
    ensure_pdf_space(pdf, 40 + len(insights) * 7)
    pdf.set_text_color(*BRAND_BLUE); pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, sanitize_for_pdf("Procurement Insights (Expanded)"), ln=True)
    pdf.set_text_color(0, 0, 0); pdf.set_font("Helvetica", "", 12)
    for ins in insights:
        ensure_pdf_space(pdf, 10)
        pdf.multi_cell(0, 7, sanitize_for_pdf(ins))
    pdf.ln(6)

    # Executive summary
    ensure_pdf_space(pdf, 40)
    pdf.set_text_color(*BRAND_BLUE); pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, sanitize_for_pdf("Executive Summary"), ln=True)
    pdf.set_text_color(0, 0, 0); pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(0, 7, sanitize_for_pdf(ai_text))
    pdf.ln(6)

    # Key Performance Metrics - compute needed height dynamically
    metrics = [
        f"Total Spend: {k['total_spend']:,.2f} {k['dominant']}",
        f"Records: {k['records']}",
        f"Vendors (Top shown): {len(k['top_v'])}",
        f"Materials (Top shown): {len(k['top_m'])}",
        f"Risk Score: {risk['score']:.1f} ({risk['band']})"
    ]
    needed = 12 + len(metrics) * 7
    ensure_pdf_space(pdf, needed)
    pdf.set_text_color(*BRAND_BLUE); pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, sanitize_for_pdf("Key Performance Metrics"), ln=True)
    pdf.set_text_color(0, 0, 0); pdf.set_font("Helvetica", "", 12)
    for m in metrics:
        ensure_pdf_space(pdf, 8)
        pdf.cell(0, 7, "- " + sanitize_for_pdf(m), ln=True)
    pdf.ln(6)

    # Critical Findings
    ensure_pdf_space(pdf, 30)
    pdf.set_text_color(*BRAND_BLUE); pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, sanitize_for_pdf("Critical Findings"), ln=True)
    pdf.set_text_color(0, 0, 0); pdf.set_font("Helvetica", "", 12)
    cf = "Top vendor concentration and multi-currency exposure create areas for negotiated savings and FX risk management. Investigate supplier consolidation and hedging strategies."
    pdf.multi_cell(0, 7, sanitize_for_pdf(cf))
    pdf.ln(6)

    # Top Performing Vendors
    topv = k.get("top_v", {})
    needed = 12 + max(1, len(topv)) * 7
    ensure_pdf_space(pdf, needed)
    pdf.set_text_color(*BRAND_BLUE); pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, sanitize_for_pdf("Top Performing Vendors"), ln=True)
    pdf.set_text_color(0, 0, 0); pdf.set_font("Helvetica", "", 12)
    if topv:
        for v, amt in topv.items():
            ensure_pdf_space(pdf, 8)
            pdf.cell(0, 7, "- " + sanitize_for_pdf(f"{v}: {amt:,.2f} {k['dominant']}"), ln=True)
    else:
        pdf.multi_cell(0, 7, sanitize_for_pdf("No vendor data available."))
    pdf.ln(6)

    # Efficiency Analysis
    eff = compute_efficiency_summary(k["df"])
    needed = 12 + (2 * 7 if eff else 1 * 7)
    ensure_pdf_space(pdf, needed)
    pdf.set_text_color(*BRAND_BLUE); pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, sanitize_for_pdf("Efficiency Analysis"), ln=True)
    pdf.set_text_color(0, 0, 0); pdf.set_font("Helvetica", "", 12)
    if eff:
        sorted_eff = sorted(eff.items(), key=lambda x: x[1]["avg_cost"])
        best, worst = sorted_eff[0], sorted_eff[-1]
        units = worst[1]["units_est"]
        gap = worst[1]["avg_cost"] - best[1]["avg_cost"]
        est_savings = units * gap if units > 0 and gap > 0 else 0.0
        pdf.multi_cell(0, 7, sanitize_for_pdf(
            f"Most Efficient: {best[0]} leads with {best[1]['avg_cost']:.2f} cost-per-unit despite {best[1]['total_spend']:,.0f} total spend."
        ))
        pdf.multi_cell(0, 7, sanitize_for_pdf(
            f"Least Efficient: {worst[0]} shows {worst[1]['avg_cost']:.2f} cost-per-unit, potential savings {est_savings:,.0f} (vendor currency)."
        ))
    else:
        pdf.multi_cell(0, 7, sanitize_for_pdf("Efficiency details unavailable."))
    pdf.ln(6)

    # Material Category Performance
    mat = compute_material_performance(k["df"])
    needed = 12 + (max(1, len(mat)) * 7)
    ensure_pdf_space(pdf, needed)
    pdf.set_text_color(*BRAND_BLUE); pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, sanitize_for_pdf("Material Category Performance"), ln=True)
    pdf.set_text_color(0, 0, 0); pdf.set_font("Helvetica", "", 12)
    if mat:
        for m, v in list(mat.items())[:10]:
            ensure_pdf_space(pdf, 8)
            pdf.cell(0, 7, "- " + sanitize_for_pdf(f"{m}: {v:,.2f} {k['dominant']}"), ln=True)
    else:
        pdf.multi_cell(0, 7, sanitize_for_pdf("No material performance data available."))
    pdf.ln(8)

    # DASHBOARD CHARTS - Start on new page
    if charts:
        pdf.add_page()
        pdf.set_text_color(*BRAND_BLUE); pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, sanitize_for_pdf("Dashboard Charts"), ln=True)
        pdf.set_text_color(0, 0, 0); pdf.ln(4)
        captions = ["Monthly Spend Trend", "Top Vendors (Bar)", "Risk Breakdown", "Material Spend Distribution"]
        for idx, ch in enumerate(charts):
            # each chart needs ~100 mm height; ensure it
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
                # if an image fails, continue safely
                continue

    # Risk Summary - ensure space
    ensure_pdf_space(pdf, 40)
    pdf.set_text_color(*BRAND_BLUE); pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, sanitize_for_pdf("Risk Summary"), ln=True)
    pdf.set_text_color(0, 0, 0); pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 7, sanitize_for_pdf(f"Risk Score: {risk['score']:.1f} ({risk['band']})"), ln=True)
    for k_, v_ in risk.get("breakdown", {}).items():
        ensure_pdf_space(pdf, 8)
        pdf.cell(0, 7, "- " + sanitize_for_pdf(f"{k_}: {v_:.1f}"), ln=True)
    pdf.ln(8)

    # AI tag at the end
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    ensure_pdf_space(pdf, 10)
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
    # read file
    if file.name.lower().endswith(".xlsx"):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)

    k = compute_kpis(df)
    risk = compute_risk(k)
    ai_text = generate_ai_text(k)

    insights = [
        currency_exposure_insight_extended(k.get("totals", {})),
        monthly_quarterly_trend_insight_extended(k.get("monthly", {})),
        material_spend_insight_extended(k.get("top_m", {}), k.get("total_spend", 0.0)),
        supplier_relationship_insight_extended(k.get("top_v", {}), k.get("total_spend", 0.0)),
        efficiency_insight_summary(compute_efficiency_summary(k["df"])),
        current_month_snapshot(k.get("monthly", {}), k.get("top_v", {}))
    ]

    # On-screen insights
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

    # Efficiency on-screen
    st.markdown("### Efficiency Analysis")
    eff_summary = compute_efficiency_summary(k["df"])
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
    mat_perf = compute_material_performance(k["df"])
    if mat_perf:
        st.dataframe(pd.DataFrame.from_dict(mat_perf, orient="index", columns=["Spend"]).sort_values("Spend", ascending=False).head(10))
    else:
        st.write("No material performance data available.")

    # Generate charts and display on screen
    st.markdown("### Dashboard Charts")
    charts = generate_dashboard_charts(k, risk)
    if charts:
        captions = []
        if k.get("monthly"): captions.append("Monthly Spend Trend")
        if k.get("top_v"): captions.append("Top Vendors")
        if risk.get("breakdown"): captions.append("Risk Breakdown")
        if k.get("top_m"): captions.append("Material Spend Distribution")
        st.image(charts, caption=captions, use_container_width=True)
    else:
        st.info("Not enough data to generate charts.")
    st.caption(AI_TAGLINE)

    # PDF generation
    if st.button("📄 Generate PDF Report"):
        pdf_buf = generate_pdf(ai_text, insights, k, risk, charts, company)
        safe_name = (company.strip().replace(" ", "_") or "Company")
        st.download_button("Download Report", pdf_buf, file_name=f"{safe_name}_Procurement_Report.pdf", mime="application/pdf")

except Exception:
    st.error("Error generating report:")
    st.text(traceback.format_exc())
