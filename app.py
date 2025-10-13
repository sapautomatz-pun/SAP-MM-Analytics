# ==========================================================
# SAP AUTOMATZ – Executive Procurement Analytics
# Version: v40 (Unicode-safe final build)
# ==========================================================

import os, io, re, datetime, math
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from openai import OpenAI
from fpdf import FPDF
from unidecode import unidecode

# ---------------- CONFIG ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"
LOGO_URL = "https://raw.githubusercontent.com/sapautomatz-pun/SAP-MM-Analytics/1d3346d7d35396f13ff06da26f24ebb5ebb70f23/sapautomatz_logo.png"
VALID_KEYS = ["SAPMM-00000000000000", "DEMO-ACCESS-12345"]

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------- STREAMLIT PAGE ----------------
st.set_page_config(
    page_title="SAP Automatz – Executive Procurement Analytics",
    page_icon="📊",
    layout="wide"
)
st.markdown("<style>.stApp header{visibility:hidden}</style>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])
with col1:
    st.image(LOGO_URL, width=140)
with col2:
    st.markdown("""
        <h2 style='margin-bottom:0;color:#1a237e;font-size:26px;'>SAP Automatz – AI Procurement Analytics</h2>
        <p style='color:#444;margin-top:0;font-size:14px;'>ERP-Compatible Executive Dashboard<br>
        <b>Automate. Analyze. Accelerate.</b></p>
    """, unsafe_allow_html=True)
st.divider()

# ---------------- ACCESS VERIFY ----------------
if "verified" not in st.session_state:
    st.session_state.verified = False

st.subheader("🔐 Verify Access Key")
key = st.text_input("Enter your access key:", type="password")

if st.button("Verify Access"):
    if key.strip() in VALID_KEYS:
        st.session_state.verified = True
        st.success("✅ Access verified successfully!")
        st.rerun()
    else:
        st.error("❌ Invalid access key. Please check and try again.")

if not st.session_state.verified:
    st.stop()

# ---------------- HELPERS ----------------
def sanitize_text(text):
    """Removes all non-encodable characters for PDF output."""
    if not text:
        return ""
    text = unidecode(str(text))
    text = text.encode("latin-1", "ignore").decode("latin-1")
    return re.sub(r'[^\x00-\x7F]+', ' ', text)

def parse_amount_and_currency(v, fallback="INR"):
    if pd.isna(v): return 0.0, fallback
    if isinstance(v, (int, float, np.number)): return float(v), fallback
    s = str(v)
    sym_map = {"₹": "INR", "Rs": "INR", "$": "USD", "USD": "USD", "€": "EUR", "EUR": "EUR"}
    cur = fallback
    for sym, c in sym_map.items():
        if sym in s:
            cur = c
            s = s.replace(sym, "")
    s = re.sub(r"[^\d.\-]", "", s)
    try:
        amt = float(s)
    except:
        amt = 0.0
    return amt, cur

def clean_dataframe(df):
    if "CURRENCY" not in df.columns:
        df["CURRENCY"] = "INR"
    amt, cur = [], []
    for _, r in df.iterrows():
        a, c = parse_amount_and_currency(r.get("AMOUNT", 0), r.get("CURRENCY", "INR"))
        amt.append(a)
        cur.append(c)
    df["AMOUNT_NUM"] = amt
    df["CURRENCY_DETECTED"] = cur
    if "VENDOR" in df.columns:
        df["VENDOR"] = df["VENDOR"].astype(str).fillna("Unknown")
    if "MATERIAL" in df.columns:
        df["MATERIAL"] = df["MATERIAL"].astype(str).fillna("Unknown")
    return df

def compute_kpis(df):
    df = clean_dataframe(df)
    if "PO_DATE" in df.columns:
        df["PO_DATE"] = pd.to_datetime(df["PO_DATE"], errors="coerce")
    totals = df.groupby("CURRENCY_DETECTED")["AMOUNT_NUM"].sum().to_dict()
    total_spend = sum(totals.values()) if totals else 0.0
    dominant = max(totals, key=totals.get) if totals else None
    top_v = df.groupby("VENDOR")["AMOUNT_NUM"].sum().nlargest(10).to_dict() if "VENDOR" in df.columns else {}
    if "QUANTITY" in df.columns:
        top_m = df.groupby("MATERIAL")["QUANTITY"].sum().nlargest(10).to_dict()
    else:
        top_m = df.groupby("MATERIAL")["AMOUNT_NUM"].sum().nlargest(10).to_dict() if "MATERIAL" in df.columns else {}
    monthly = {}
    if "PO_DATE" in df.columns:
        d = df.dropna(subset=["PO_DATE"])
        if not d.empty:
            d["YM"] = d["PO_DATE"].dt.to_period("M").astype(str)
            monthly = d.groupby("YM")["AMOUNT_NUM"].sum().to_dict()
    return {
        "totals": totals,
        "total_spend": total_spend,
        "dominant": dominant,
        "top_v": top_v,
        "top_m": top_m,
        "monthly": monthly,
        "records": len(df),
        "df": df
    }

def compute_procurement_risk(df, k):
    df_local = k.get("df", df)
    totals = k.get("totals", {})
    total_spend = k.get("total_spend", 0.0)
    v = df_local.groupby("VENDOR")["AMOUNT_NUM"].sum()
    nv = v.size if not v.empty else 0
    top_share = (v.max()/total_spend) if total_spend and not v.empty else 1.0
    v_conc = max(0.0, (1.0 - top_share)) * 100
    v_div = min(100.0, (nv / 50) * 100)
    if totals and total_spend:
        dom = k.get("dominant")
        dom_share = totals.get(dom, 0.0)/total_spend if dom else 1.0
        c_expo = dom_share * 100
    else:
        c_expo = 100.0
    mvals = list(k.get("monthly", {}).values())
    if len(mvals) >= 3 and np.mean(mvals) > 0:
        cv = np.std(mvals)/(np.mean(mvals)+1e-9)
        m_vol = max(0.0, 1-min(cv, 2))*100
    else:
        m_vol = 80.0
    score = v_conc*0.25 + v_div*0.25 + c_expo*0.25 + m_vol*0.25
    score = float(max(0.0, min(100.0, score)))
    band = "Low" if score >= 67 else ("Medium" if score >= 34 else "High")
    return {
        "score": score,
        "band": band,
        "breakdown": {
            "Vendor Concentration": v_conc,
            "Vendor Diversity": v_div,
            "Currency Exposure": c_expo,
            "Monthly Volatility": m_vol
        }
    }

def generate_ai(k):
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a procurement analytics expert."},
                {"role": "user", "content": f"Provide executive insights and key recommendations for this procurement dataset:\n{k}"}
            ],
            temperature=0.2, max_tokens=800
        )
        return sanitize_text(r.choices[0].message.content)
    except Exception as e:
        return sanitize_text(f"AI Error: {e}")

# ---------------- PDF ----------------
class PDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, f"SAP Automatz Confidential | Page {self.page_no()} of {{nb}}", 0, 0, "C")

def generate_pdf(ai_text, kpis, charts, company, summary_text, risk):
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "", 11)

    # Cover Page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, sanitize_text("Executive Procurement Analysis Report"), ln=True, align="C")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, sanitize_text(f"Prepared for: {company}"), ln=True, align="C")
    pdf.cell(0, 8, f"Generated on: {datetime.date.today().strftime('%d %B %Y')}", ln=True, align="C")
    pdf.ln(15)
    pdf.multi_cell(0, 7, sanitize_text(summary_text))
    pdf.image(LOGO_URL, x=160, y=260, w=30)

    # Insights Page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "AI-Generated Insights", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, sanitize_text(ai_text))

    # Risk Breakdown
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Procurement Risk Breakdown", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for kx, vx in risk["breakdown"].items():
        pdf.cell(0, 8, f"{kx}: {vx:,.2f}", ln=True)

    return io.BytesIO(pdf.output(dest="S").encode("latin-1", "ignore"))

# ---------------- MAIN APP ----------------
st.title("📊 Executive Procurement Dashboard")
company = st.text_input("Enter Company Name:", "ABC Manufacturing Pvt Ltd")
f = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx"])
if not f:
    st.stop()

df = pd.read_excel(f) if f.name.endswith(".xlsx") else pd.read_csv(f)
k = compute_kpis(df)
risk = compute_procurement_risk(df, k)
ai = generate_ai(k)

charts = []
summary = ai[:1000]
pdf = generate_pdf(ai, k, charts, company, summary, risk)

st.download_button("📄 Download Full Executive Report",
                   pdf, "SAP_Automatz_Executive_Report.pdf",
                   "application/pdf")
