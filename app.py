# ==========================================================
# SAP AUTOMATZ - Procurement Analytics AI (v27.6 FINAL BUILD)
# ==========================================================
# ✅ FIX: Chart plotting error (auto numeric clean)
# ✅ FIX: PDF blank issue resolved
# ✅ Includes Executive Summary, KPIs, Charts
# ==========================================================

import os, io, re, datetime, platform, requests, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from openai import OpenAI
from fpdf import FPDF
from unidecode import unidecode

# -------------------------
# CONFIG
# -------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "https://sapautomatz-backend.onrender.com")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"
LOGO_URL = "https://raw.githubusercontent.com/sapautomatz-pun/SAP-MM-Analytics/1d3346d7d35396f13ff06da26f24ebb5ebb70f23/sapautomatz_logo.png"

# Fonts
if platform.system() == "Windows":
    FONT_PATH = "./fonts/DejaVuSans.ttf"
    FONT_PATH_BOLD = "./fonts/DejaVuSans-Bold.ttf"
else:
    FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    FONT_PATH_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------
# STREAMLIT CONFIG
# -------------------------
st.set_page_config(page_title="SAP Automatz - Procurement Analytics AI", page_icon="📊", layout="wide")
st.markdown("<style>.stApp header{visibility:hidden}</style>", unsafe_allow_html=True)

col1, col2 = st.columns([1,3])
with col1:
    st.image(LOGO_URL, width=160)
with col2:
    st.markdown("<h2 style='margin-bottom:0'>SAP Automatz Procurement Analytics AI</h2>"
                "<p style='color:#555;margin-top:0'>Automate. Analyze. Accelerate 🚀</p>", unsafe_allow_html=True)
st.divider()

# -------------------------
# ACCESS CONTROL
# -------------------------
if "verified" not in st.session_state:
    st.session_state.verified = False

if not st.session_state.verified:
    st.markdown("### 🔐 Verify Access")
    access_key = st.text_input("Enter access key", type="password")
    if st.button("Verify Access"):
        try:
            resp = requests.get(f"{BACKEND_URL}/verify_key/{access_key}", timeout=25)
            if resp.status_code == 200:
                j = resp.json()
                if j.get("valid"):
                    st.session_state.verified = True
                    st.session_state.access_key = access_key
                    st.success(f"✅ Access verified (valid till {j.get('expiry_date')})")
                    st.rerun()
                else:
                    st.error(f"❌ Invalid access key: {j.get('reason','check again')}")
            else:
                st.error("Backend did not respond properly.")
        except Exception as e:
            st.error(f"Verification error: {e}")
    st.stop()

# -------------------------
# HELPERS
# -------------------------
def sanitize_text(text):
    if text is None:
        return ""
    text = unidecode(str(text))
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", "", text)
    return text.strip()

def clean_numeric(series):
    """Convert mixed currency/strings to numeric"""
    return (
        series.astype(str)
        .replace(r"[^\d.\-]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
        .fillna(0)
    )

def calculate_kpis(df):
    # Auto-detect numeric field
    num_cols = [c for c in df.columns if any(x in c.upper() for x in ["VALUE","AMOUNT","TOTAL","PRICE","COST"])]
    if not num_cols:
        df["AMOUNT"] = 0
    else:
        df["AMOUNT"] = clean_numeric(df[num_cols[0]])
    df["CURRENCY_DETECTED"] = "INR"
    kpis = {
        "records": len(df),
        "total_spend": float(df["AMOUNT"].sum()),
    }
    if "PO_DATE" in df.columns:
        start, end = pd.to_datetime(df["PO_DATE"], errors="coerce").min(), pd.to_datetime(df["PO_DATE"], errors="coerce").max()
        if pd.notna(start) and pd.notna(end):
            kpis["date_range"] = f"{start.strftime('%d-%b-%Y')} to {end.strftime('%d-%b-%Y')}"
    if "VENDOR" in df.columns:
        try:
            kpis["top_vendor"] = df.groupby("VENDOR")["AMOUNT"].sum().idxmax()
        except:
            kpis["top_vendor"] = "N/A"
    return kpis

def ai_summary(k):
    prompt = f"""Summarize SAP procurement insights for:
Records: {k['records']}
Total Spend: {k['total_spend']:,}
Date Range: {k.get('date_range','N/A')}
Top Vendor: {k.get('top_vendor','N/A')}
Write 3 short paragraphs about performance, vendor dependency, and optimization opportunities."""
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": "You are a SAP procurement analyst writing concise business insights."},
                      {"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=500)
        return sanitize_text(r.choices[0].message.content)
    except Exception as e:
        return f"AI Error: {e}"

# -------------------------
# PDF
# -------------------------
class PDF(FPDF):
    def header(self): pass
    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "", 9)
        self.set_text_color(130,130,130)
        self.cell(0,10,"© 2025 SAP Automatz – Powered by Gen AI",align="C")

def generate_pdf(ai_text, k, charts, customer, key):
    pdf = PDF()
    pdf.add_font("DejaVu","",FONT_PATH,uni=True)
    pdf.add_font("DejaVu","B",FONT_PATH_BOLD,uni=True)

    # --- COVER PAGE ---
    pdf.add_page()
    pdf.set_fill_color(33,86,145)
    pdf.rect(0,0,210,297,'F')
    try: pdf.image(LOGO_URL,70,30,70)
    except: pass
    pdf.set_text_color(255,255,255)
    pdf.set_font("DejaVu","B",22)
    pdf.ln(110)
    pdf.cell(0,10,"Procurement Analytics Report",align="C",ln=True)
    pdf.ln(10)
    pdf.set_font("DejaVu","",14)
    pdf.cell(0,10,f"Customer: {customer}",align="C",ln=True)
    pdf.cell(0,10,f"Access Key: {key}",align="C",ln=True)
    pdf.cell(0,10,f"Date: {datetime.date.today().strftime('%d %b %Y')}",align="C",ln=True)

    # --- EXECUTIVE SUMMARY ---
    pdf.add_page()
    pdf.set_text_color(0,0,0)
    pdf.set_font("DejaVu","B",14)
    pdf.cell(0,10,"Executive Summary",ln=True)
    pdf.ln(4)
    pdf.set_font("DejaVu","",11)

    ai_text = sanitize_text(ai_text)
    if not ai_text or ai_text.startswith("AI Error"):
        pdf.multi_cell(0,8,"No AI summary generated. Please check your API key or connection.")
    else:
        for line in ai_text.split("\n"):
            if line.strip():
                pdf.multi_cell(0,7,line)
                pdf.ln(2)

    # --- KPIs ---
    pdf.ln(6)
    pdf.set_font("DejaVu","B",12)
    pdf.cell(0,8,"Key Performance Indicators",ln=True)
    pdf.set_font("DejaVu","",11)
    for kx,v in k.items():
        pdf.multi_cell(0,6,f"{kx}: {v}")

    # --- CHARTS ---
    for path in charts:
        if path and os.path.exists(path):
            pdf.add_page()
            pdf.set_font("DejaVu","B",12)
            pdf.cell(0,8,os.path.basename(path).replace("_"," ").title(),ln=True)
            try:
                pdf.image(path,w=160)
            except:
                pdf.multi_cell(0,6,"Chart unavailable.")

    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output

# -------------------------
# MAIN APP
# -------------------------
st.title("📊 Procurement Analytics Dashboard (v27.6)")
file = st.file_uploader("📂 Upload SAP Procurement Data", type=["csv","xlsx"])

if file:
    df = pd.read_excel(file) if file.name.endswith(".xlsx") else pd.read_csv(file)

    k = calculate_kpis(df)

    charts=[]
    st.subheader("🏢 Vendor Spend Overview")
    if "VENDOR" in df.columns and "AMOUNT" in df.columns:
        vendor = df.groupby("VENDOR")["AMOUNT"].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8,4))
        vendor.plot(kind="bar", ax=ax, color="#2E86C1")
        ax.set_ylabel("Spend")
        ax.set_title("Top 10 Vendors by Spend")
        vendor_chart = "vendor_chart.png"
        fig.tight_layout()
        fig.savefig(vendor_chart)
        charts.append(vendor_chart)
        st.pyplot(fig)

    st.subheader("📦 Material Spend Distribution")
    if "MATERIAL" in df.columns and "AMOUNT" in df.columns:
        mat = df.groupby("MATERIAL")["AMOUNT"].sum().sort_values(ascending=False).head(10)
        fig2, ax2 = plt.subplots(figsize=(6,6))
        ax2.pie(mat, labels=mat.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title("Top 10 Materials by Spend")
        mat_chart = "material_chart.png"
        fig2.tight_layout()
        fig2.savefig(mat_chart)
        charts.append(mat_chart)
        st.pyplot(fig2)

    ai_text = ai_summary(k)
    st.subheader("AI Insights Summary")
    st.markdown(ai_text)

    pdf_bytes = generate_pdf(ai_text, k, charts, "ABC Manufacturing Pvt Ltd", st.session_state.access_key)
    st.download_button("📄 Download Full Report PDF", pdf_bytes, f"SAP_Report_{datetime.date.today()}.pdf", "application/pdf")
