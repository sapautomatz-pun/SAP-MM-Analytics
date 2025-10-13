# app.py
# SAP Automatz – Procurement Analytics v42_production_final
# Stable baseline version (used yesterday before KPI enhancement)

import os
import io
import re
import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from fpdf import FPDF

# Optional: OpenAI + Unicode
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

try:
    from unidecode import unidecode
    def clean_text(s): return unidecode(str(s))
except Exception:
    def clean_text(s): return str(s)

# ---------------- CONFIG ----------------
MODEL = "gpt-4o-mini"
LOGO_URL = "https://raw.githubusercontent.com/sapautomatz-pun/SAP-MM-Analytics/1d3346d7d35396f13ff06da26f24ebb5ebb70f23/sapautomatz_logo.png"
VALID_KEYS = {"SAPMM-00000000000000", "DEMO-ACCESS-12345"}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_AVAILABLE and OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None
    OPENAI_AVAILABLE = False


# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="SAP Automatz – Procurement Analytics", layout="wide")
st.markdown("<style>.stApp header{visibility:hidden}</style>", unsafe_allow_html=True)

col_logo, col_title = st.columns([1, 3])
with col_logo:
    st.image(LOGO_URL, width=120)
with col_title:
    st.markdown(
        "<h2 style='color:#1a237e;margin-bottom:0'>SAP Automatz – Procurement Analytics</h2>"
        "<p style='margin-top:0;color:#555;'>Automate. Analyze. Accelerate.</p>",
        unsafe_allow_html=True,
    )
st.divider()


# ---------------- HELPERS ----------------
def sanitize_text_for_pdf(text):
    if not text:
        return ""
    s = clean_text(str(text))
    s = s.replace("•", "-").replace("—", "-").replace("–", "-")
    return s.encode("latin-1", "ignore").decode("latin-1")


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
        val = float(s)
    except:
        val = 0.0
    return val, detected


def prepare_dataframe(df: pd.DataFrame):
    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]
    if "AMOUNT" not in df.columns:
        for c in df.columns:
            if "AMT" in c.upper():
                df.rename(columns={c: "AMOUNT"}, inplace=True)
    if "CURRENCY" not in df.columns:
        df["CURRENCY"] = "INR"
    amounts, currencies = [], []
    for _, row in df.iterrows():
        a, c = parse_amount_and_currency(row.get("AMOUNT", 0), row.get("CURRENCY", "INR"))
        amounts.append(a)
        currencies.append(c)
    df["AMOUNT_NUM"] = amounts
    df["CURRENCY_DETECTED"] = currencies
    return df


def compute_kpis(df):
    df = prepare_dataframe(df)
    totals = df.groupby("CURRENCY_DETECTED")["AMOUNT_NUM"].sum().to_dict()
    total_spend = sum(totals.values())
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
            "top_v": top_v, "top_m": top_m, "monthly": monthly, "records": len(df), "df": df}


def compute_risk(k):
    df = k.get("df", pd.DataFrame())
    total_spend = k["total_spend"]
    v = df.groupby("VENDOR")["AMOUNT_NUM"].sum() if "VENDOR" in df.columns else pd.Series()
    nv = len(v)
    top_share = (v.max() / total_spend) if total_spend else 1
    v_conc = (1 - top_share) * 100
    v_div = min(100, nv / 50 * 100)
    dom = k["dominant"]
    c_expo = (k["totals"].get(dom, 0) / total_spend) * 100 if total_spend else 100
    mvals = list(k["monthly"].values())
    m_vol = 100 * (1 - np.std(mvals) / (np.mean(mvals) + 1e-9)) if len(mvals) > 2 else 80
    score = (v_conc + v_div + c_expo + m_vol) / 4
    band = "Low" if score >= 67 else ("Medium" if score >= 34 else "High")
    return {"score": score, "band": band, "breakdown": {
        "Vendor Concentration": v_conc,
        "Vendor Diversity": v_div,
        "Currency Exposure": c_expo,
        "Monthly Volatility": m_vol
    }}


def generate_ai_text(k):
    base_summary = f"Total spend: {k['total_spend']:.2f}, Dominant currency: {k['dominant']}, Top vendors: {list(k['top_v'].keys())[:3]}"
    if OPENAI_AVAILABLE and client:
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a procurement analytics expert."},
                    {"role": "user", "content": f"Generate an executive summary and recommendations for:\n{base_summary}"}
                ],
                temperature=0.3,
                max_tokens=400
            )
            return sanitize_text_for_pdf(r.choices[0].message.content)
        except:
            pass
    return sanitize_text_for_pdf(
        f"Executive Insights:\n• Total spend: {k['total_spend']:.2f} ({k['dominant']})\n"
        f"• Key vendors: {', '.join(list(k['top_v'].keys())[:3]) or 'N/A'}\n"
        "Recommendations:\n• Negotiate better rates with top vendors.\n"
        "• Optimize inventory for high-demand materials.\n"
        "• Review currency exposure and diversify vendor base."
    )


# ---------------- PDF GENERATION ----------------
class PDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"SAP Automatz | Page {self.page_no()}", 0, 0, "C")


def generate_pdf(ai_text, k, charts, company, risk):
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Procurement Analytics Report", ln=True, align="C")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Company: {company}", ln=True, align="C")
    pdf.cell(0, 10, f"Date: {datetime.date.today().strftime('%d-%b-%Y')}", ln=True, align="C")
    pdf.ln(10)
    pdf.multi_cell(0, 7, ai_text)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Risk Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, f"Risk Score: {risk['score']:.0f} ({risk['band']})")
    for ch in charts:
        pdf.add_page()
        pdf.image(ch, x=20, y=30, w=170)
    out = io.BytesIO(pdf.output(dest="S").encode("latin-1", "ignore"))
    out.seek(0)
    return out


# ---------------- APP FLOW ----------------
st.subheader("🔐 Verify Access Key")
col1, col2 = st.columns([3, 1])
with col1:
    access_key = st.text_input("Enter access key", type="password")
with col2:
    if st.button("Verify"):
        if access_key.strip() in VALID_KEYS:
            st.session_state["verified"] = True
            st.success("Access verified successfully.")
            st.rerun()
        else:
            st.error("Invalid key. Please try again.")

if not st.session_state.get("verified"):
    st.stop()

st.markdown("### Upload Procurement File")
file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
if not file:
    st.info("Upload your procurement extract to continue.")
    st.stop()

df = pd.read_excel(file) if file.name.endswith(".xlsx") else pd.read_csv(file)
k = compute_kpis(df)
risk = compute_risk(k)
ai_text = generate_ai_text(k)

# ---------------- CHARTS ----------------
charts = []
plt.figure(figsize=(6, 4))
plt.pie(k["totals"].values(), labels=k["totals"].keys(), autopct="%1.1f%%")
plt.title("Currency Distribution")
plt.savefig("chart1.png", bbox_inches="tight")
charts.append("chart1.png")

plt.figure(figsize=(7, 4))
plt.bar(k["top_v"].keys(), k["top_v"].values(), color="#1976D2")
plt.xticks(rotation=45, ha="right")
plt.title("Top Vendors")
plt.savefig("chart2.png", bbox_inches="tight")
charts.append("chart2.png")

plt.figure(figsize=(7, 4))
plt.bar(k["top_m"].keys(), k["top_m"].values(), color="#43A047")
plt.xticks(rotation=45, ha="right")
plt.title("Top Materials")
plt.savefig("chart3.png", bbox_inches="tight")
charts.append("chart3.png")

# ---------------- DISPLAY ----------------
st.markdown("## Executive Dashboard")
st.metric("Total Spend", f"{k['total_spend']:,.2f} {k['dominant']}")
st.metric("Risk Score", f"{risk['score']:.0f} ({risk['band']})")
st.markdown("### AI Insights")
st.write(ai_text)
st.image(charts, use_container_width=True)

if st.button("📄 Generate PDF Report"):
    pdf = generate_pdf(ai_text, k, charts, "ABC Manufacturing Pvt Ltd", risk)
    st.download_button("Download Report", pdf, file_name="SAP_Automatz_Report.pdf", mime="application/pdf")
