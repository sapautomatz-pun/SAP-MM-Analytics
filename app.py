# ==========================================================
# SAP AUTOMATZ - Procurement Analytics AI (v26.4)
# ==========================================================
# Fixes:
# ✅ FPDF "Not enough horizontal space" error
# ✅ Full Unicode-safe DejaVu font for all PDF text
# ✅ Removes emojis/special chars before rendering
# ==========================================================

import os, io, re, json, base64, datetime, platform, requests, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from openai import OpenAI
from fpdf import FPDF

# -------------------------
# CONFIG
# -------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "https://sapautomatz-backend.onrender.com")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAILERSEND_API_KEY = os.getenv("MAILERSEND_API_KEY")
MODEL = "gpt-4o-mini"
LOGO_URL = "https://raw.githubusercontent.com/sapautomatz-pun/SAP-MM-Analytics/1d3346d7d35396f13ff06da26f24ebb5ebb70f23/sapautomatz_logo.png"

# Font paths (DejaVu ensures full UTF-8 support)
if platform.system() == "Windows":
    FONT_PATH = "./fonts/DejaVuSans.ttf"
    FONT_PATH_BOLD = "./fonts/DejaVuSans-Bold.ttf"
else:
    FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    FONT_PATH_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

client = OpenAI(api_key=OPENAI_API_KEY)

# Streamlit Setup
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
# ACCESS VERIFICATION
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

def clean_text_for_pdf(text):
    """Remove emojis and unrenderable characters safely"""
    if text is None: return ""
    text = str(text)
    # Remove emojis and non-BMP characters
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]', '', text)
    return text.strip()

def parse_value_and_currency(val, default_currency=None):
    if pd.isna(val): return (np.nan, default_currency)
    if isinstance(val,(int,float,np.number)): return (float(val), default_currency)
    s = str(val).strip()
    m = re.match(r"^([^\d\-\+]+)\s*([0-9,.\-]+)$", s)
    if m:
        sym, num = m.group(1).strip(), m.group(2).replace(",", "")
        try: num=float(num)
        except: num=np.nan
        code=CURRENCY_SYMBOLS.get(sym,None)
        return(num,code)
    for sym,code in CURRENCY_SYMBOLS.items():
        if sym in s:
            num=re.sub(r"[^\d.\-]","",s)
            try:return(float(num),code)
            except:return(np.nan,code)
    try:return(float(s.replace(",","")),default_currency)
    except:return(np.nan,default_currency)

def detect_currency_symbol(df):
    header=" ".join(df.columns).upper()
    mapping={"INR":"₹","USD":"$","EUR":"€","GBP":"£","JPY":"¥","CNY":"¥"}
    for code,sym in mapping.items():
        if code in header:return sym
    for col in df.columns:
        if df[col].dtype=="object":
            sample=" ".join(df[col].astype(str).head(5).values)
            for sym in mapping.values():
                if sym in sample:return sym
    return "₹"

def calculate_kpis_and_parse(df):
    if "CURRENCY" not in df.columns:
        df["CURRENCY"]=None
    inferred_symbol=detect_currency_symbol(df)
    inferred_default=CURRENCY_SYMBOLS.get(inferred_symbol,None)
    amounts,codes=[],[]
    for _,row in df.iterrows():
        raw=row.get("VALUE",None)
        default=row.get("CURRENCY",None) or inferred_default
        num,code=parse_value_and_currency(raw,default)
        try:num=float(num)
        except:num=0.0
        amounts.append(num)
        codes.append(code or inferred_default or "USD")
    df["AMOUNT"]=pd.Series(amounts,dtype=float)
    df["CURRENCY_DETECTED"]=codes
    kpis={
        "records":len(df),
        "sums_per_currency":df.groupby("CURRENCY_DETECTED")["AMOUNT"].sum().to_dict(),
        "total_spend_raw":float(np.nansum(df["AMOUNT"]))
    }
    if "PO_DATE" in df.columns and "GRN_DATE" in df.columns:
        df["CYCLE_DAYS"]=(df["GRN_DATE"]-df["PO_DATE"]).dt.days
        kpis["avg_cycle_days"]=round(df["CYCLE_DAYS"].mean(skipna=True),1)
        kpis["delayed_count"]=int(df[df["CYCLE_DAYS"]>7].shape[0])
    else:
        kpis["avg_cycle_days"],kpis["delayed_count"]=None,0
    try:kpis["top_vendor"]=df.groupby("VENDOR")["AMOUNT"].sum().idxmax()
    except:kpis["top_vendor"]="N/A"
    return df,kpis

def ai_summary(prompt):
    try:
        r=client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content":"You are a SAP procurement analyst writing executive summaries."},
                      {"role":"user","content":prompt}],
            temperature=0.3,max_tokens=700)
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"AI error: {e}"

# -------------------------
# PDF UTILITIES
# -------------------------
class PDF(FPDF):
    def header(self):
        self.set_fill_color(33,86,145)
        self.rect(0,0,210,20,'F')
        try:self.image(LOGO_URL,10,2,25)
        except:pass
        self.set_text_color(255,255,255)
        self.set_font("DejaVu","B",14)
        self.cell(0,10,"SAP Automatz - Procurement Analytics Report",align="C",ln=True)
    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu","I",9)
        self.set_text_color(130,130,130)
        self.cell(0,10,"© 2025 SAP Automatz – Powered by Gen AI",align="C")

def generate_pdf(ai_text,kpis,chart_path,currency):
    pdf=PDF()
    pdf.add_font("DejaVu","",FONT_PATH,uni=True)
    pdf.add_font("DejaVu","B",FONT_PATH_BOLD,uni=True)
    pdf.set_font("DejaVu","",11)
    pdf.add_page()

    # Clean text before rendering
    ai_text = clean_text_for_pdf(ai_text)

    pdf.cell(0,10,"Executive Summary",ln=True)
    pdf.ln(4)
    for line in ai_text.split("\n"):
        try:
            pdf.multi_cell(0,6,clean_text_for_pdf(line))
        except:
            pdf.multi_cell(0,6,"[Content Omitted - Unreadable Characters]")
    pdf.ln(5)
    pdf.set_font("DejaVu","B",12)
    pdf.cell(0,8,"Key Performance Indicators",ln=True)
    pdf.set_font("DejaVu","",11)
    for k,v in kpis.items():
        text = clean_text_for_pdf(f"{k}: {v}")
        try: pdf.multi_cell(0,6,text)
        except: pdf.multi_cell(0,6,"[Invalid Text Skipped]")
    if chart_path and os.path.exists(chart_path):
        pdf.add_page()
        pdf.cell(0,8,"Vendor Spend Distribution",ln=True)
        try: pdf.image(chart_path,w=160)
        except: pdf.multi_cell(0,6,"Chart unavailable.")
    return io.BytesIO(pdf.output(dest="S").encode("latin-1","ignore"))

# -------------------------
# MAIN APP
# -------------------------
st.title("📊 Procurement Analytics Dashboard (v26.4)")
file=st.file_uploader("📂 Upload SAP Procurement Data",type=["csv","xlsx"])

if file:
    df=pd.read_excel(file) if file.name.endswith(".xlsx") else pd.read_csv(file)
    df=normalize_columns(coerce_types(df))
    df,k=calculate_kpis_and_parse(df)
    currency=detect_currency_symbol(df)

    st.subheader("Vendor Spend Overview")
    vendor_sum=df.groupby("VENDOR")["AMOUNT"].sum().sort_values(ascending=False).head(10)
    fig,ax=plt.subplots(figsize=(8,4))
    vendor_sum.plot(kind="bar",ax=ax,color="#2E86C1")
    ax.set_ylabel("Spend")
    ax.set_title("Top 10 Vendors by Spend")
    chart_path="vendor_spend_chart.png"
    fig.tight_layout(); fig.savefig(chart_path)
    st.pyplot(fig)

    st.subheader("Data Summary Preview")
    top_vendors=(vendor_sum/vendor_sum.sum()*100).reset_index()
    top_vendors.columns=["Vendor","Spend %"]
    st.dataframe(top_vendors)

    prompt=f"Records: {k['records']}, Totals: {k['sums_per_currency']}, Avg Cycle Days: {k['avg_cycle_days']}, Delayed: {k['delayed_count']}, Top Vendor: {k['top_vendor']}"
    ai_text=ai_summary(prompt)
    st.subheader("AI Insights Summary")
    st.markdown(ai_text)

    pdf_bytes=generate_pdf(ai_text,k,chart_path,currency)
    st.download_button("📄 Download Executive Dashboard PDF",pdf_bytes,f"SAP_Report_{datetime.date.today()}.pdf","application/pdf")
