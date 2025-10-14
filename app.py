# app.py
# SAP Automatz – Procurement Analytics v10
# Fixes UnboundLocalError in compute_risk + adds logo & tagline on all PDF pages

import os
import io
import re
import datetime
import tempfile
import traceback
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from fpdf import FPDF

# ---------- CONFIG ----------
MODEL = "gpt-4o-mini"
LOGO_PATH = "sapautomatz_logo.png"  # your uploaded logo
TAGLINE = "Automate. Analyze. Accelerate."
VALID_KEYS = {"SAPMM-00000000000000", "DEMO-ACCESS-12345"}

# ---------- STREAMLIT CONFIG ----------
st.set_page_config(page_title="SAP Automatz – Procurement Analytics", layout="wide")
st.markdown("<style>.stApp header{visibility:hidden}</style>", unsafe_allow_html=True)

col_logo, col_title = st.columns([1, 3])
with col_logo:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=110)
with col_title:
    st.markdown(
        "<h2 style='color:#1a237e;margin-bottom:0'>SAP Automatz – Procurement Analytics</h2>"
        f"<p style='margin-top:0;color:#555;'>{TAGLINE}</p>",
        unsafe_allow_html=True,
    )
st.divider()

st.session_state.setdefault("verified", False)

# ---------- HELPERS ----------
def sanitize_text_for_pdf(text):
    return str(text or "").encode("latin-1", "ignore").decode("latin-1")

def parse_amount_and_currency(value, fallback="INR"):
    sym_map = {"₹": "INR", "Rs": "INR", "$": "USD", "€": "EUR", "£": "GBP"}
    if pd.isna(value):
        return 0.0, fallback
    s, detected = str(value), fallback
    for sym, code in sym_map.items():
        if sym in s:
            detected, s = code, s.replace(sym, "")
    s = re.sub(r"[^\d.\-]", "", s)
    try:
        return float(s or 0), detected
    except Exception:
        return 0.0, detected

def prepare_dataframe(df: pd.DataFrame):
    df.columns = [c.strip().upper() for c in df.columns]
    if "AMOUNT" not in df.columns:
        for c in df.columns:
            if "AMT" in c.upper():
                df.rename(columns={c: "AMOUNT"}, inplace=True)
                break
    if "CURRENCY" not in df.columns:
        df["CURRENCY"] = "INR"
    df["AMOUNT_NUM"], df["CURRENCY_DETECTED"] = zip(*df.apply(lambda r: parse_amount_and_currency(r.get("AMOUNT", 0), r.get("CURRENCY", "INR")), axis=1))
    return df

def compute_kpis(df):
    df = prepare_dataframe(df)
    totals = df.groupby("CURRENCY_DETECTED")["AMOUNT_NUM"].sum().to_dict() if "CURRENCY_DETECTED" in df.columns else {}
    total_spend = sum(totals.values()) if totals else float(df["AMOUNT_NUM"].sum())
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
            "top_v": top_v, "top_m": top_m, "monthly": monthly, "df": df}

def compute_risk(k):
    df = k.get("df", pd.DataFrame())
    total = k.get("total_spend", 0.0) or 0.0
    if total == 0 or df.empty:
        return {"score": 0, "band": "Low", "breakdown": {}}

    # safely set dominant currency
    dom = k.get("dominant", "INR")
    totals = k.get("totals", {})
    v = df.groupby("VENDOR")["AMOUNT_NUM"].sum() if "VENDOR" in df.columns else pd.Series(dtype=float)
    nv = len(v)
    top_share = float(v.max()) / total if not v.empty else 1.0
    v_conc = (1 - top_share) * 100
    v_div = min(100.0, (nv / 50.0) * 100.0) if nv > 0 else 0.0
    c_expo = (totals.get(dom, 0.0) / total) * 100.0 if total else 100.0
    mvals = list(k.get("monthly", {}).values())
    m_vol = 100.0 * (1 - np.std(mvals) / (np.mean(mvals) + 1e-9)) if len(mvals) > 2 and np.mean(mvals) != 0 else 80.0
    score = float(np.clip((v_conc + v_div + c_expo + m_vol) / 4.0, 0.0, 100.0))
    band = "Low" if score >= 67 else ("Medium" if score >= 34 else "High")
    return {"score": score, "band": band, "breakdown": {
        "Vendor Concentration": v_conc,
        "Vendor Diversity": v_div,
        "Currency Exposure": c_expo,
        "Monthly Volatility": m_vol
    }}

def compute_efficiency(df):
    if "VENDOR" not in df.columns or df.empty:
        return {}, {}
    eff = df.groupby("VENDOR")["AMOUNT_NUM"].mean().sort_values(ascending=False)
    return eff.head(3).to_dict(), eff.tail(3).to_dict()

def compute_material_performance(df):
    if "MATERIAL" not in df.columns or df.empty:
        return {}
    mat_perf = df.groupby("MATERIAL")["AMOUNT_NUM"].sum().sort_values(ascending=False).head(10)
    return mat_perf.to_dict()

# ---------- PDF CLASS ----------
class PDF(FPDF):
    def header(self):
        if os.path.exists(LOGO_PATH):
            self.image(LOGO_PATH, 10, 8, 20)
        self.set_xy(32, 10)
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 6, "SAP Automatz", ln=True)
        self.set_xy(32, 16)
        self.set_font("Helvetica", "I", 9)
        self.cell(0, 5, TAGLINE, ln=True)
        self.ln(8)
        self.line(10, 25, 200, 25)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"SAP Automatz | Page {self.page_no()}", 0, 0, "C")

# ---------- CHARTS ----------
def generate_dashboard_charts(k, risk):
    chart_files = []
    if k.get("monthly"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            plt.figure(figsize=(6, 3))
            plt.plot(list(k["monthly"].keys()), list(k["monthly"].values()), marker="o")
            plt.title("Monthly Spend Trend")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(tmp.name, bbox_inches="tight")
            chart_files.append(tmp.name)
        plt.close()
    if k.get("top_v"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            top5 = dict(list(k["top_v"].items())[:5])
            plt.figure(figsize=(6, 3))
            plt.bar(top5.keys(), top5.values())
            plt.title("Top 5 Vendors by Spend")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(tmp.name, bbox_inches="tight")
            chart_files.append(tmp.name)
        plt.close()
    if risk.get("breakdown"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            plt.figure(figsize=(6, 3))
            plt.bar(risk["breakdown"].keys(), risk["breakdown"].values())
            plt.title("Risk Breakdown")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(tmp.name, bbox_inches="tight")
            chart_files.append(tmp.name)
        plt.close()
    return chart_files

# ---------- PDF GENERATOR ----------
def generate_pdf(k, company, risk):
    pdf = PDF()
    pdf.alias_nb_pages()

    # Cover Page
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, "Procurement Analytics Report", ln=True, align="C")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 14)
    pdf.cell(0, 10, f"Company: {company}", ln=True, align="C")
    pdf.cell(0, 10, f"Generated On: {datetime.date.today().strftime('%d-%b-%Y')}", ln=True, align="C")

    # Dashboard charts
    dash_charts = generate_dashboard_charts(k, risk)
    for ch in dash_charts:
        pdf.add_page()
        pdf.image(ch, x=20, y=40, w=170)

    # Risk Summary
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Risk Summary", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Risk Score: {risk.get('score',0):.0f} ({risk.get('band')})", ln=True)
    pdf.ln(4)
    for k_, v_ in risk.get("breakdown", {}).items():
        pdf.cell(0, 8, f"- {k_}: {v_:.1f}", ln=True)

    # Output
    out_bytes = pdf.output(dest="S").encode("latin-1", "ignore")
    out = io.BytesIO(out_bytes)
    out.seek(0)
    return out

# ---------- APP FLOW ----------
st.subheader("🔐 Verify Access Key")
col1, col2 = st.columns([3, 1])
with col1:
    key = st.text_input("Enter access key", type="password")
with col2:
    if st.button("Verify"):
        if key.strip() in VALID_KEYS:
            st.session_state["verified"] = True
            st.success("Access verified.")
            st.rerun()
        else:
            st.error("Invalid key.")

if not st.session_state["verified"]:
    st.stop()

file = st.file_uploader("Upload Procurement File (CSV or Excel)", type=["csv", "xlsx"])
if not file:
    st.info("Please upload a procurement dataset.")
    st.stop()

company = st.text_input("Enter your Company Name", "ABC Manufacturing Pvt Ltd")

try:
    df = pd.read_excel(file) if file.name.endswith(".xlsx") else pd.read_csv(file)
    k = compute_kpis(df)
    risk = compute_risk(k)

    st.metric("Total Spend", f"{k['total_spend']:,.2f} {k['dominant']}")
    st.metric("Risk Score", f"{risk['score']:.0f} ({risk['band']})")

    if st.button("📄 Generate PDF Report"):
        pdf = generate_pdf(k, company, risk)
        st.download_button("Download Report", pdf, file_name=f"{company.replace(' ','_')}_Report.pdf", mime="application/pdf")
except Exception:
    st.error("Error generating report:")
    st.text(traceback.format_exc())
