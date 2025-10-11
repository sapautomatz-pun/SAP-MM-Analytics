# ==========================================================
# SAP AUTOMATZ - Procurement Analytics AI (v27.3 ENTERPRISE)
# ==========================================================
# FINAL ENTERPRISE RELEASE
# ✅ AI summary + KPIs visible in PDF
# ✅ Bulletproof text sanitization (UTF-8 safe)
# ✅ Layout optimization for multipage PDF
# ✅ Retains charts + cover page
# ==========================================================

import os, io, re, datetime, platform, requests, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from openai import OpenAI
from fpdf import FPDF
from unidecode import unidecode  # NEW import for safe AI text cleaning

# -------------------------
# CONFIG
# -------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "https://sapautomatz-backend.onrender.com")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"
LOGO_URL = "https://raw.githubusercontent.com/sapautomatz-pun/SAP-MM-Analytics/1d3346d7d35396f13ff06da26f24ebb5ebb70f23/sapautomatz_logo.png"

# Fonts (DejaVu for PDF)
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
CURRENCY_SYMBOLS = {"₹":"INR", "$":"USD", "€":"EUR", "£":"GBP", "¥":"JPY"}

def normalize_columns(df):
    df = df.rename(columns=lambda x: str(x).strip().upper())
    mapping = {"PO NO":"PO_NUMBER","PURCHASE ORDER":"PO_NUMBER","PO_DATE":"PO_DATE","GRN_DATE":"GRN_DATE",
               "VENDOR":"VENDOR","SUPPLIER":"VENDOR","MATERIAL":"MATERIAL","QUANTITY":"QUANTITY",
               "VALUE":"VALUE","AMOUNT":"VALUE","CURRENCY":"CURRENCY"}
    for k,v in mapping.items():
        for col in df.columns:
            if k in col:
                df.rename(columns={col:v}, inplace=True)
    return df

def coerce_types(df):
    if "PO_DATE" in df.columns:
        df["PO_DATE"] = pd.to_datetime(df["PO_DATE"], errors="coerce")
    if "GRN_DATE" in df.columns:
        df["GRN_DATE"] = pd.to_datetime(df["GRN_DATE"], errors="coerce")
    if "QUANTITY" in df.columns:
        df["QUANTITY"] = pd.to_numeric(df["QUANTITY"], errors="coerce")
    return df

def sanitize_text(text):
    """Clean & encode text for PDF output"""
    if text is None: return ""
    text = unidecode(str(text))
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]', '', text)
    return text.strip()

def calculate_kpis_and_parse(df):
    if "CURRENCY" not in df.columns:
        df["CURRENCY"]=None
    inferred_symbol="₹"
    inferred_default=CURRENCY_SYMBOLS.get(inferred_symbol,"INR")
    amounts,codes=[],[]
    for _,row in df.iterrows():
        raw=row.get("VALUE",None)
        try:num=float(str(raw).replace(",",""))
        except:num=0.0
        amounts.append(num)
        codes.append(inferred_default)
    df["AMOUNT"]=pd.Series(amounts,dtype=float)
    df["CURRENCY_DETECTED"]=codes
    kpis={
        "records":len(df),
        "sums_per_currency":df.groupby("CURRENCY_DETECTED")["AMOUNT"].sum().to_dict(),
        "total_spend_raw":float(np.nansum(df["AMOUNT"]))
    }
    if "PO_DATE" in df.columns:
        start,end=df["PO_DATE"].min(),df["PO_DATE"].max()
        kpis["date_range"]=f"{start.strftime('%d-%b-%Y')} to {end.strftime('%d-%b-%Y')}" if pd.notna(start) else "N/A"
    else:kpis["date_range"]="N/A"
    if "PO_DATE" in df.columns and "GRN_DATE" in df.columns:
        df["CYCLE_DAYS"]=(df["GRN_DATE"]-df["PO_DATE"]).dt.days
        kpis["avg_cycle_days"]=round(df["CYCLE_DAYS"].mean(skipna=True),1)
        kpis["delayed_count"]=int(df[df["CYCLE_DAYS"]>7].shape[0])
    else:
        kpis["avg_cycle_days"],kpis["delayed_count"]=None,0
    try:kpis["top_vendor"]=df.groupby("VENDOR")["AMOUNT"].sum().idxmax()
    except:kpis["top_vendor"]="N/A"
    return df,kpis

def ai_summary(k):
    """Generate structured AI summary"""
    prompt=f"""Generate a clear executive summary for a SAP Procurement Analytics report:
- Records: {k['records']}
- Total spend by currency: {k['sums_per_currency']}
- Date Range: {k['date_range']}
- Avg Cycle Days: {k['avg_cycle_days']}
- Delayed Orders: {k['delayed_count']}
- Top Vendor: {k['top_vendor']}
Write 3 short paragraphs explaining procurement performance, efficiency and improvement opportunities."""
    try:
        r=client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content":"You are an SAP Procurement Analyst summarizing data-driven insights."},
                      {"role":"user","content":prompt}],
            temperature=0.3,max_tokens=500)
        return sanitize_text(r.choices[0].message.content)
    except Exception as e:
        return f"AI error: {e}"

# -------------------------
# PDF
# -------------------------
class PDF(FPDF):
    def header(self): pass
    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu","",9)
        self.set_text_color(130,130,130)
        self.cell(0,10,"© 2025 SAP Automatz – Powered by Gen AI",align="C")

def add_cover(pdf, customer, key):
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
    pdf.ln(15)
    pdf.set_font("DejaVu","",11)
    pdf.cell(0,10,"Confidential: For authorized SAP Automatz users only",align="C",ln=True)

def generate_pdf(ai_text,k,charts,customer,key):
    pdf=PDF()
    pdf.add_font("DejaVu","",FONT_PATH,uni=True)
    pdf.add_font("DejaVu","B",FONT_PATH_BOLD,uni=True)
    add_cover(pdf,customer,key)
    # --- Summary Page ---
    pdf.add_page()
    pdf.set_font("DejaVu","B",14)
    pdf.cell(0,10,"Executive Summary",ln=True)
    pdf.set_font("DejaVu","",11)
    pdf.ln(4)
    for para in ai_text.split("\n"):
        pdf.multi_cell(0,6,sanitize_text(para))
    pdf.ln(6)
    pdf.set_font("DejaVu","B",12)
    pdf.cell(0,8,"Key Performance Indicators",ln=True)
    pdf.set_font("DejaVu","",11)
    for kx,v in k.items():
        pdf.multi_cell(0,6,f"{kx}: {v}")
    # --- Charts ---
    for path in charts:
        if path and os.path.exists(path):
            pdf.add_page()
            pdf.set_font("DejaVu","B",12)
            pdf.cell(0,8,os.path.basename(path).replace("_"," ").title(),ln=True)
            try: pdf.image(path,w=160)
            except: pdf.multi_cell(0,6,"Chart unavailable.")
    return io.BytesIO(pdf.output(dest="S").encode("latin-1","ignore"))

# -------------------------
# MAIN APP
# -------------------------
st.title("📊 Procurement Analytics Dashboard (v27.3)")
file=st.file_uploader("📂 Upload SAP Procurement Data",type=["csv","xlsx"])

if file:
    df=pd.read_excel(file) if file.name.endswith(".xlsx") else pd.read_csv(file)
    df=normalize_columns(coerce_types(df))
    df,k=calculate_kpis_and_parse(df)

    # Charts
    charts=[]
    st.subheader("🏢 Vendor Spend Overview")
    vendor=df.groupby("VENDOR")["AMOUNT"].sum().sort_values(ascending=False).head(10)
    fig,ax=plt.subplots(figsize=(8,4))
    vendor.plot(kind="bar",ax=ax,color="#2E86C1")
    ax.set_ylabel("Spend"); ax.set_title("Top 10 Vendors by Spend")
    vchart="vendor_chart.png"; fig.tight_layout(); fig.savefig(vchart)
    charts.append(vchart); st.pyplot(fig)

    st.subheader("📦 Material Spend Distribution")
    mat=df.groupby("MATERIAL")["AMOUNT"].sum().sort_values(ascending=False).head(10)
    fig2,ax2=plt.subplots(figsize=(6,6))
    ax2.pie(mat,labels=mat.index,autopct='%1.1f%%',startangle=90)
    ax2.set_title("Top 10 Materials by Spend")
    mchart="material_chart.png"; fig2.tight_layout(); fig2.savefig(mchart)
    charts.append(mchart); st.pyplot(fig2)

    # AI Summary
    ai_text=ai_summary(k)
    st.subheader("AI Insights Summary")
    st.markdown(ai_text)

    # Generate PDF
    pdf_bytes=generate_pdf(ai_text,k,charts,"ABC Manufacturing Pvt Ltd",st.session_state.access_key)
    st.download_button("📄 Download Full Report PDF",pdf_bytes,f"SAP_Report_{datetime.date.today()}.pdf","application/pdf")
