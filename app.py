# ==========================================================
# SAP AUTOMATZ - Procurement Analytics AI App (v17.0)
# ==========================================================
# Features:
#  ✅ Secure Access Key Verification + Remember Me
#  ✅ Branded Header (always visible)
#  ✅ GPT-4o Insights, PDF Reports
#  ✅ Airtable Storage + Customer Report Portal
# ==========================================================

import os
import io
import re
import json
import requests
import pandas as pd
import datetime
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openai import OpenAI
from fpdf import FPDF

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------
BACKEND_URL = "https://sapautomatz-backend.onrender.com"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAILERSEND_API_KEY = os.getenv("MAILERSEND_API_KEY")
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME", "Reports")

MAILER_FROM_EMAIL = "sapautomatz@gmail.com"
MAILER_FROM_NAME = "SAP Automatz"
MODEL = "gpt-4o-mini"

LOGO_URL = "https://raw.githubusercontent.com/sapautomatz-pun/SAP-MM-Analytics/1d3346d7d35396f13ff06da26f24ebb5ebb70f23/sapautomatz_logo.png"

client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------------------------------------
# STREAMLIT SETTINGS
# ----------------------------------------------------------
st.set_page_config(page_title="SAP Automatz - Procurement Analytics AI", page_icon="📊", layout="wide")

# CSS Styling
st.markdown("""
<style>
.main {background-color: #F6F9FC;}
div.block-container {padding-top: 1.5rem;}
h1, h2, h3, h4, h5 {color: #215691;}
.stApp header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# HEADER
# ----------------------------------------------------------
col1, col2 = st.columns([1, 3])
with col1:
    st.image(LOGO_URL, width=160)
with col2:
    st.markdown("""
        <h2 style='margin-bottom:0;'>SAP Automatz Procurement Analytics AI</h2>
        <p style='color:#444;margin-top:0;'>Automate. Analyze. Accelerate 🚀</p>
    """, unsafe_allow_html=True)
st.divider()

# ----------------------------------------------------------
# SESSION STATE (Remember Me)
# ----------------------------------------------------------
if "access_verified" not in st.session_state:
    st.session_state.access_verified = False
if "customer_name" not in st.session_state:
    st.session_state.customer_name = ""
if "plan_type" not in st.session_state:
    st.session_state.plan_type = ""
if "expiry" not in st.session_state:
    st.session_state.expiry = ""

# ----------------------------------------------------------
# ACCESS VERIFICATION
# ----------------------------------------------------------
if not st.session_state.access_verified:
    st.markdown("### 🔐 Step 1: Verify Your Access Key")
    access_key = st.text_input("Enter your Access Key (received after payment)", type="password")
    remember = st.checkbox("Remember me (for this device)")

    if st.button("Verify Access"):
        try:
            res = requests.get(f"{BACKEND_URL}/verify_access", params={"key": access_key}, timeout=10)
            data = res.json()
            if res.status_code == 200 and data.get("status") == "ok":
                st.session_state.access_verified = True
                st.session_state.customer_name = data.get("name")
                st.session_state.plan_type = data.get("plan").capitalize()
                st.session_state.expiry = data.get("expiry")
                if remember:
                    st.session_state["remember_me"] = True
                st.success(f"✅ Access granted to **{data.get('name')}** ({data.get('plan').capitalize()} plan, valid till {data.get('expiry')})")
                st.rerun()
            else:
                st.error("❌ Invalid or expired Access Key.")
        except Exception as e:
            st.error(f"Error verifying access: {e}")
    st.stop()

# ----------------------------------------------------------
# POST-VERIFICATION DASHBOARD
# ----------------------------------------------------------
st.sidebar.image(LOGO_URL, width=150)
page = st.sidebar.radio("📘 Navigate", ["🔍 Analyze & Generate", "📬 My Reports Portal"])
st.sidebar.caption("Automate. Analyze. Accelerate 🚀")

# Display session details
st.success(f"✅ Logged in as {st.session_state.customer_name} ({st.session_state.plan_type} plan, valid till {st.session_state.expiry})")

# ----------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------
def normalize_columns(df):
    df = df.rename(columns=lambda x: str(x).strip().upper())
    mapping = {
        "PO NO": "PO_NUMBER", "PURCHASE ORDER": "PO_NUMBER",
        "PO_DATE": "PO_DATE", "GRN_DATE": "GRN_DATE",
        "VENDOR": "VENDOR", "SUPPLIER": "VENDOR",
        "MATERIAL": "MATERIAL", "QUANTITY": "QUANTITY",
        "VALUE": "VALUE", "AMOUNT": "VALUE"
    }
    for key, val in mapping.items():
        for col in df.columns:
            if key in col:
                df.rename(columns={col: val}, inplace=True)
    return df

def coerce_types(df):
    for c in ["PO_DATE", "GRN_DATE"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["QUANTITY", "VALUE"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def calculate_kpis(df):
    kpis = {}
    kpis["records"] = len(df)
    kpis["total_spend"] = float(df["VALUE"].sum()) if "VALUE" in df.columns else 0
    if "PO_DATE" in df.columns and "GRN_DATE" in df.columns:
        df["CYCLE_DAYS"] = (df["GRN_DATE"] - df["PO_DATE"]).dt.days
        kpis["avg_cycle"] = round(df["CYCLE_DAYS"].mean(), 1)
        kpis["delays"] = len(df[df["CYCLE_DAYS"] > 7])
    else:
        kpis["avg_cycle"], kpis["delays"] = None, 0
    if "VENDOR" in df.columns and "VALUE" in df.columns:
        kpis["top_vendor"] = df.groupby("VENDOR")["VALUE"].sum().idxmax()
    else:
        kpis["top_vendor"] = "N/A"
    return kpis, df

def build_prompt(kpis):
    return f"""
Analyze these KPIs:
Total Records: {kpis['records']}
Total Spend: ₹{kpis['total_spend']:,}
Average Cycle Time: {kpis['avg_cycle']} days
Delayed Shipments: {kpis['delays']}
Top Vendor: {kpis['top_vendor']}
Provide concise insights under:
1️⃣ Executive Summary
2️⃣ Root Causes
3️⃣ Recommendations
4️⃣ KPIs for next month
"""

def generate_ai_summary(prompt):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a senior SAP procurement analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=600
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating AI summary: {e}"

class BrandedPDF(FPDF):
    def header(self):
        self.set_fill_color(33, 86, 145)
        self.rect(0, 0, 210, 20, 'F')
        self.image(LOGO_URL, 10, 2, 25)
        self.set_text_color(255,255,255)
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "SAP Automatz - Procurement Analytics Report", align="C", ln=True)
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(130,130,130)
        self.cell(0, 10, "© 2025 SAP Automatz – Powered by Gen AI", align="C")

def sanitize_text(text):
    return re.sub(r'[^\x20-\x7E]+', ' ', str(text or ""))[:200]

def generate_pdf(ai_text, kpis):
    pdf = BrandedPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 10, f"Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", ln=True)
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Key KPIs", ln=True)
    pdf.set_font("Helvetica", size=11)
    for k, v in kpis.items():
        pdf.multi_cell(0, 8, f"{sanitize_text(k).title()}: {sanitize_text(v)}")
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "AI Insight Summary", ln=True)
    pdf.set_font("Helvetica", size=11)
    for line in sanitize_text(ai_text).split(". "):
        pdf.multi_cell(0, 6, line.strip())
    return io.BytesIO(bytes(pdf.output(dest='S').encode('latin-1', errors='ignore')))

# ----------------------------------------------------------
# PAGE: ANALYZE & GENERATE
# ----------------------------------------------------------
if page == "🔍 Analyze & Generate":
    uploaded_file = st.file_uploader("📁 Upload your SAP Procurement Data", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)
        df = normalize_columns(coerce_types(df))
        kpis, df = calculate_kpis(df)
        ai_text = generate_ai_summary(build_prompt(kpis))
        st.markdown(ai_text)
        pdf_bytes = generate_pdf(ai_text, kpis)
        st.download_button("📄 Download Report", pdf_bytes, f"SAP_Report_{datetime.date.today()}.pdf", "application/pdf")

# ----------------------------------------------------------
# PAGE: CUSTOMER REPORT PORTAL
# ----------------------------------------------------------
if page == "📬 My Reports Portal":
    st.markdown("### 📦 Access Your Reports")
    email = st.text_input("Enter your registered email")
    if st.button("🔍 Fetch Reports"):
        endpoint = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}?filterByFormula=FIND('{email}',{{Email}})"
        headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
        r = requests.get(endpoint, headers=headers)
        if r.status_code == 200:
            data = r.json()
            reports = [{"Report Name": rec["fields"].get("Report Name"), "Report URL": rec["fields"].get("Report URL"), "Created On": rec["fields"].get("Created On")} for rec in data.get("records", [])]
            if reports:
                for r in reports:
                    st.markdown(f"📄 **{r['Report Name']}** — _{r['Created On']}_  \n [🔗 Download]({r['Report URL']})", unsafe_allow_html=True)
            else:
                st.info("No reports found for this email.")
        else:
            st.error("Error fetching from Airtable.")
