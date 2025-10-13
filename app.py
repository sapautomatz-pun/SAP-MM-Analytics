# app.py
# SAP Automatz – v41a (Stable Release with st.rerun fix)
# Full functional release with Verify Access, dashboard, AI insights, and PDF generation.

import os
import io
import re
import datetime
import math
from typing import Dict, Any

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from fpdf import FPDF

# Optional: import OpenAI if available
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# Optional unidecode for safer text encoding
try:
    from unidecode import unidecode
    def _unidecode(s): return unidecode(s)
except Exception:
    def _unidecode(s): return s


# ---------------- CONFIG ----------------
MODEL = "gpt-4o-mini"
LOGO_URL = "https://raw.githubusercontent.com/sapautomatz-pun/SAP-MM-Analytics/1d3346d7d35396f13ff06da26f24ebb5ebb70f23/sapautomatz_logo.png"
VALID_KEYS = {"SAPMM-00000000000000", "DEMO-ACCESS-12345"}  # Replace with your real access keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or None
if OPENAI_AVAILABLE and OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None
    OPENAI_AVAILABLE = False


# ---------------- STREAMLIT UI CONFIG ----------------
st.set_page_config(page_title="SAP Automatz – Procurement Analytics", page_icon="📊", layout="wide")
st.markdown("<style>.stApp header{visibility:hidden}</style>", unsafe_allow_html=True)

# Header
col_logo, col_head = st.columns([1, 3])
with col_logo:
    st.image(LOGO_URL, width=140)
with col_head:
    st.markdown(
        "<h2 style='margin:0;color:#1a237e'>SAP Automatz – Procurement Analytics</h2>"
        "<div style='color:#444;font-size:13px'>Automate. Analyze. Accelerate.</div>",
        unsafe_allow_html=True
    )
st.divider()


# ---------------- SESSION STATE ----------------
for key in ["verified", "uploaded", "df", "kpis", "risk", "ai_text"]:
    if key not in st.session_state:
        st.session_state[key] = False if key in ["verified", "uploaded"] else None


# ---------------- HELPERS ----------------
def sanitize_text_for_pdf(text: str) -> str:
    """Clean and sanitize text for FPDF compatibility."""
    if text is None:
        return ""
    s = _unidecode(str(text))
    s = s.replace("•", "-").replace("—", "-").replace("–", "-")
    s = s.encode("latin-1", "ignore").decode("latin-1")
    s = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", " ", s)
    return s


def parse_amount_and_currency(raw_value, fallback="INR"):
    sym_map = {"₹": "INR", "Rs": "INR", "$": "USD", "€": "EUR", "GBP": "GBP"}
    if pd.isna(raw_value):
        return 0.0, fallback
    if isinstance(raw_value, (int, float, np.number)):
        return float(raw_value), fallback
    s = str(raw_value)
    detected = fallback
    for sym, code in sym_map.items():
        if sym in s:
            detected = code
            s = s.replace(sym, "")
    s = re.sub(r"[^\d\.\-]", "", s)
    try:
        val = float(s)
    except:
        val = 0.0
    return val, detected


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]
    if "AMOUNT" not in df.columns:
        for c in df.columns:
            if "AMOUNT" in c.upper():
                df.rename(columns={c: "AMOUNT"}, inplace=True)
                break
    if "CURRENCY" not in df.columns:
        df["CURRENCY"] = np.nan
    amounts, currencies = [], []
    for _, row in df.iterrows():
        a, cur = parse_amount_and_currency(row.get("AMOUNT", 0), row.get("CURRENCY", "INR"))
        amounts.append(a)
        currencies.append(cur)
    df["AMOUNT_NUM"] = amounts
    df["CURRENCY_DETECTED"] = currencies
    if "VENDOR" not in df.columns:
        for c in df.columns:
            if "VENDOR" in c.upper():
                df.rename(columns={c: "VENDOR"}, inplace=True)
                break
    if "MATERIAL" not in df.columns:
        for c in df.columns:
            if "MATERIAL" in c.upper():
                df.rename(columns={c: "MATERIAL"}, inplace=True)
                break
    df["VENDOR"] = df.get("VENDOR", "Unknown")
    df["MATERIAL"] = df.get("MATERIAL", "Unknown")
    return df


def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    df = clean_dataframe(df)
    if "PO_DATE" in df.columns:
        df["PO_DATE"] = pd.to_datetime(df["PO_DATE"], errors="coerce")
    totals = df.groupby("CURRENCY_DETECTED")["AMOUNT_NUM"].sum().to_dict()
    total_spend = sum(totals.values())
    dominant = max(totals, key=totals.get) if totals else "INR"
    top_v = df.groupby("VENDOR")["AMOUNT_NUM"].sum().nlargest(10).to_dict()
    top_m = df.groupby("MATERIAL")["AMOUNT_NUM"].sum().nlargest(10).to_dict()
    monthly = {}
    if "PO_DATE" in df.columns:
        temp = df.dropna(subset=["PO_DATE"])
        if not temp.empty:
            temp["MONTH"] = temp["PO_DATE"].dt.to_period("M").astype(str)
            monthly = temp.groupby("MONTH")["AMOUNT_NUM"].sum().to_dict()
    return {"totals": totals, "total_spend": total_spend, "dominant": dominant,
            "top_v": top_v, "top_m": top_m, "monthly": monthly, "df": df, "records": len(df)}


def compute_risk(k: Dict[str, Any]) -> Dict[str, Any]:
    df = k["df"]
    total_spend = k["total_spend"]
    v = df.groupby("VENDOR")["AMOUNT_NUM"].sum()
    nv = len(v)
    top_share = (v.max() / total_spend) if total_spend else 1.0
    v_conc = (1.0 - top_share) * 100
    v_div = min(100.0, nv / 50 * 100)
    dom = k["dominant"]
    c_expo = (k["totals"].get(dom, 0) / total_spend) * 100 if total_spend else 100
    mvals = list(k["monthly"].values())
    m_vol = 100 * (1 - np.std(mvals) / (np.mean(mvals) + 1e-9)) if len(mvals) >= 3 else 80
    score = (v_conc + v_div + c_expo + m_vol) / 4
    band = "Low" if score >= 67 else ("Medium" if score >= 34 else "High")
    return {"score": score, "band": band, "breakdown": {
        "Vendor Concentration": v_conc,
        "Vendor Diversity": v_div,
        "Currency Exposure": c_expo,
        "Monthly Volatility": m_vol
    }}


def generate_ai_text(k: Dict[str, Any]) -> str:
    base_summary = f"Total spend: {k['total_spend']:.2f}\nCurrencies: {list(k['totals'].keys())}\nTop Vendors: {list(k['top_v'].keys())[:3]}"
    if OPENAI_AVAILABLE and client:
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a procurement analyst."},
                    {"role": "user", "content": f"Generate concise procurement summary:\n{base_summary}"}
                ],
                temperature=0.2, max_tokens=400
            )
            return sanitize_text_for_pdf(r.choices[0].message.content)
        except:
            pass
    return sanitize_text_for_pdf(f"Executive Summary:\n{base_summary}\n\nRecommendations:\n- Negotiate with key vendors\n- Monitor INR currency exposure")


# ---------------- CHART HELPERS ----------------
def safe_pie(values, labels, path, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    if not values or sum(values) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.set_title(title)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def safe_bar(labels, values, path, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    if not values:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        ax.bar(labels, values, color="#1E88E5")
        ax.set_title(title)
        plt.xticks(rotation=45, ha="right")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def safe_line(x, y, path, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    if not y:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        ax.plot(x, y, marker="o")
        ax.set_title(title)
        plt.xticks(rotation=45, ha="right")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------- PDF ----------------
class PDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"SAP Automatz | Page {self.page_no()}", 0, 0, "C")


def generate_pdf(ai_text, k, charts, company, summary, risk):
    pdf = PDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Procurement Analysis Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Prepared for: {company}", ln=True, align="C")
    pdf.cell(0, 10, f"Date: {datetime.date.today().strftime('%d-%b-%Y')}", ln=True, align="C")
    pdf.ln(20)
    pdf.multi_cell(0, 8, sanitize_text_for_pdf(summary))
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "AI Insights", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, sanitize_text_for_pdf(ai_text))
    for c in charts:
        if os.path.exists(c):
            pdf.add_page()
            pdf.image(c, x=20, y=40, w=170)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"Risk Index: {risk['score']:.0f} ({risk['band']})", ln=True)
    out = io.BytesIO(pdf.output(dest="S").encode("latin-1", "ignore"))
    out.seek(0)
    return out


# ---------------- MAIN APP ----------------
st.subheader("🔐 Verify Access Key")
col1, col2 = st.columns([3, 1])
with col1:
    access_key = st.text_input("Enter access key", type="password")
with col2:
    if st.button("Verify"):
        if access_key.strip() in VALID_KEYS:
            st.session_state.verified = True
            st.success("Access verified successfully.")
            st.rerun()  # ✅ fixed from experimental_rerun
        else:
            st.error("Invalid key. Try again.")

if not st.session_state.verified:
    st.stop()

company = st.text_input("Company Name", value="ABC Manufacturing Pvt Ltd")
file = st.file_uploader("Upload Procurement File", type=["csv", "xlsx"])
if not file:
    st.info("Upload a CSV or Excel file to continue.")
    st.stop()

df = pd.read_excel(file) if file.name.endswith(".xlsx") else pd.read_csv(file)
k = compute_kpis(df)
risk = compute_risk(k)
ai = generate_ai_text(k)

# Charts
charts = []
safe_pie(list(k["totals"].values()), list(k["totals"].keys()), "chart_currency.png", "Currency Distribution")
safe_bar(list(k["top_v"].keys()), list(k["top_v"].values()), "chart_vendor.png", "Top Vendors by Spend")
safe_bar(list(k["top_m"].keys()), list(k["top_m"].values()), "chart_material.png", "Top Materials by Spend")
safe_line(list(k["monthly"].keys()), list(k["monthly"].values()), "chart_monthly.png", "Monthly Trend")
charts = ["chart_currency.png", "chart_vendor.png", "chart_material.png", "chart_monthly.png"]

# Dashboard
st.metric("Total Spend", f"{k['total_spend']:,.2f} {k['dominant']}")
st.metric("Risk Score", f"{risk['score']:.0f} ({risk['band']})")
st.write(ai)
st.image(charts, use_container_width=True)

if st.button("Generate Report PDF"):
    pdf = generate_pdf(ai, k, charts, company, ai[:600], risk)
    st.download_button("📄 Download PDF Report", pdf, "SAP_Automatz_Report.pdf", mime="application/pdf")
