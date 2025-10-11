# ==========================================================
# SAP AUTOMATZ - Procurement Analytics AI App (v20.0)
# ==========================================================
# ✅ Color-coded KPI boxes
# ✅ Safe PDF generation (fpdf2)
# ✅ GPT-4o AI insights, charts, Airtable portal
# ==========================================================

import os, io, re, json, datetime, requests, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from openai import OpenAI
from fpdf import FPDF

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------
BACKEND_URL = "https://sapautomatz-backend.onrender.com"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME", "Reports")
MODEL = "gpt-4o-mini"
LOGO_URL = "https://raw.githubusercontent.com/sapautomatz-pun/SAP-MM-Analytics/1d3346d7d35396f13ff06da26f24ebb5ebb70f23/sapautomatz_logo.png"

client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------------------------------------
# STREAMLIT CONFIG
# ----------------------------------------------------------
st.set_page_config(page_title="SAP Automatz - Procurement Analytics AI", page_icon="📊", layout="wide")
st.markdown("<style>.stApp header{visibility:hidden}</style>", unsafe_allow_html=True)

# HEADER
c1, c2 = st.columns([1,3])
with c1: st.image(LOGO_URL, width=160)
with c2:
    st.markdown("<h2 style='margin-bottom:0'>SAP Automatz Procurement Analytics AI</h2>"
                "<p style='color:#555;margin-top:0'>Automate. Analyze. Accelerate 🚀</p>", unsafe_allow_html=True)
st.divider()

# ----------------------------------------------------------
# ACCESS CHECK
# ----------------------------------------------------------
if "access_verified" not in st.session_state:
    st.session_state.access_verified = False

if not st.session_state.access_verified:
    key = st.text_input("Enter your Access Key", type="password")
    if st.button("Verify"):
        try:
            r = requests.get(f"{BACKEND_URL}/verify_access", params={"key": key}, timeout=10)
            data = r.json()
            if r.status_code==200 and data.get("status")=="ok":
                st.session_state.access_verified=True
                st.session_state.customer_name=data["name"]
                st.session_state.plan=data["plan"]
                st.session_state.expiry=data["expiry"]
                st.success(f"✅ Welcome {data['name']} ({data['plan']} plan, valid till {data['expiry']})")
                st.rerun()
            else: st.error("Invalid or expired key")
        except Exception as e: st.error(e)
    st.stop()

# SIDEBAR
st.sidebar.image(LOGO_URL, width=150)
page = st.sidebar.radio("📘 Navigate", ["🔍 Analyze & Generate", "📬 My Reports Portal"])

# ----------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------
def normalize_columns(df):
    df = df.rename(columns=lambda x: str(x).strip().upper())
    mapping = {"PO NO":"PO_NUMBER","PURCHASE ORDER":"PO_NUMBER","PO_DATE":"PO_DATE","GRN_DATE":"GRN_DATE",
               "VENDOR":"VENDOR","SUPPLIER":"VENDOR","MATERIAL":"MATERIAL","QUANTITY":"QUANTITY",
               "VALUE":"VALUE","AMOUNT":"VALUE"}
    for k,v in mapping.items():
        for col in df.columns:
            if k in col: df.rename(columns={col:v}, inplace=True)
    return df

def coerce_types(df):
    for c in ["PO_DATE","GRN_DATE"]: 
        if c in df.columns: df[c]=pd.to_datetime(df[c],errors="coerce")
    for c in ["QUANTITY","VALUE"]: 
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
    return df

def calc_kpis(df):
    k={"records":len(df),
       "total_spend":float(df["VALUE"].sum()) if "VALUE" in df else 0}
    if "PO_DATE" in df and "GRN_DATE" in df:
        df["CYCLE_DAYS"]=(df["GRN_DATE"]-df["PO_DATE"]).dt.days
        k["avg_cycle"]=round(df["CYCLE_DAYS"].mean(),1)
        k["delays"]=len(df[df["CYCLE_DAYS"]>7])
    else: k["avg_cycle"],k["delays"]=None,0
    if "VENDOR" in df and "VALUE" in df:
        k["top_vendor"]=df.groupby("VENDOR")["VALUE"].sum().idxmax()
    else: k["top_vendor"]="N/A"
    return k,df

def build_prompt(k):
    return f"""Analyze KPIs:
Total Records: {k['records']}
Total Spend: ₹{k['total_spend']:,}
Average Cycle Time: {k['avg_cycle']} days
Delayed Shipments: {k['delays']}
Top Vendor: {k['top_vendor']}
Create an executive summary with sections:
1️⃣ Executive Summary
2️⃣ Key Observations
3️⃣ Root Causes
4️⃣ Recommendations
5️⃣ Next Month’s KPI Goals
"""

def ai_summary(prompt):
    try:
        r=client.chat.completions.create(model=MODEL,messages=[
            {"role":"system","content":"You are an expert SAP procurement analyst preparing a professional summary."},
            {"role":"user","content":prompt}],temperature=0.3,max_tokens=700)
        return r.choices[0].message.content.strip()
    except Exception as e: return f"AI error: {e}"

def charts(df):
    paths={}
    if "VENDOR" in df and "VALUE" in df:
        top=df.groupby("VENDOR")["VALUE"].sum().nlargest(10)
        plt.figure(figsize=(8,4));top.plot(kind="bar",color="steelblue")
        plt.title("Top 10 Vendors by Spend");plt.tight_layout()
        p="/tmp/vendors.png";plt.savefig(p);paths["Top Vendors"]=p;st.image(p)
    if "MATERIAL" in df and "VALUE" in df:
        sp=df.groupby("MATERIAL")["VALUE"].sum().nlargest(6)
        plt.figure(figsize=(5,5));plt.pie(sp,labels=sp.index,autopct="%1.1f%%");plt.title("Material Spend");p="/tmp/mat.png"
        plt.savefig(p);paths["Material Spend"]=p;st.image(p)
    if "PO_DATE" in df and "VALUE" in df:
        df["MONTH"]=df["PO_DATE"].dt.to_period("M").astype(str)
        m=df.groupby("MONTH")["VALUE"].sum()
        plt.figure(figsize=(8,4));m.plot(marker="o",color="orange")
        plt.title("Monthly Spend Trend");plt.tight_layout()
        p="/tmp/trend.png";plt.savefig(p);paths["Monthly Trend"]=p;st.image(p)
    return paths

# ----------------------------------------------------------
# PDF GENERATION (safe + color-coded KPI boxes)
# ----------------------------------------------------------
class PDF(FPDF):
    def header(self):
        self.set_fill_color(33,86,145)
        self.rect(0,0,210,20,'F')
        try:self.image(LOGO_URL,10,2,25)
        except:pass
        self.set_text_color(255,255,255);self.set_font("Helvetica","B",14)
        self.cell(0,10,"SAP Automatz - Procurement Analytics Report",align="C",ln=True)
    def footer(self):
        self.set_y(-15);self.set_font("Helvetica","I",9)
        self.set_text_color(130,130,130)
        self.cell(0,10,"©2025 SAP Automatz – Powered by Gen AI",align="C")

def kpi_color(value,thresholds):
    """Return (r,g,b) based on thresholds = (good,warning)"""
    try:v=float(value)
    except:return(180,180,180)
    good,warning=thresholds
    if v<=good:return(120,200,120)
    elif v<=warning:return(255,210,80)
    else:return(255,100,100)

def generate_pdf(ai_text,k,chart_paths):
    def clean(t):
        if t is None or (isinstance(t,float) and pd.isna(t)):return"N/A"
        t=str(t).replace("₹","Rs ");t=re.sub(r"[^\x20-\x7E]"," ",t)
        if len(t)>100:t=t[:97]+"..."
        return t.strip()
    pdf=PDF();pdf.add_page()
    pdf.set_font("Helvetica","B",16)
    pdf.cell(0,10,"📈 Executive Summary Dashboard",ln=True)
    pdf.ln(6)

    # KPI Boxes (color coded)
    metrics=[("Total Spend (₹)",k["total_spend"],(100000,500000)),
             ("Avg Cycle Time (Days)",k["avg_cycle"],(7,15)),
             ("Delayed Shipments",k["delays"],(10,30))]
    x0=15;y=40;w=60;h=20
    for label,val,thr in metrics:
        r,g,b=kpi_color(val,thr)
        pdf.set_fill_color(r,g,b)
        pdf.rect(x0,y,w,h,"F")
        pdf.set_xy(x0+2,y+2)
        pdf.set_text_color(0,0,0)
        pdf.set_font("Helvetica","B",11)
        pdf.multi_cell(w-4,6,f"{label}\n{clean(val)}",align="C")
        x0+=65
    pdf.ln(35)

    pdf.set_font("Helvetica","B",14);pdf.cell(0,8,"💼 Executive Summary",ln=True)
    pdf.set_font("Helvetica",size=11)
    sections={"Key Observations":"🔍","Root Causes":"⚙️","Recommendations":"🧭","Next Month’s KPI Goals":"🎯"}
    for sec,emo in sections.items():
        pdf.set_font("Helvetica","B",12)
        pdf.cell(0,8,f"{emo} {sec}",ln=True)
        pdf.set_font("Helvetica",size=11)
        match=re.findall(f"{sec}(.*?)(?=\n[A-Z]|$)",ai_text,re.IGNORECASE|re.DOTALL)
        text=clean(match[0]) if match else "N/A"
        try:pdf.multi_cell(0,6,text)
        except:pdf.multi_cell(0,6,text[:80]+"...")
        pdf.ln(3)

    pdf.add_page()
    for name,path in chart_paths.items():
        pdf.set_font("Helvetica","B",12);pdf.cell(0,8,name,ln=True)
        try:pdf.image(path,w=160)
        except:pdf.multi_cell(0,6,f"(Chart {name} missing)")
        pdf.ln(6)

    return io.BytesIO(pdf.output(dest="S").encode("latin-1","ignore"))

# ----------------------------------------------------------
# PAGE LOGIC
# ----------------------------------------------------------
if page=="🔍 Analyze & Generate":
    file=st.file_uploader("Upload SAP Procurement Data",type=["csv","xlsx"])
    if file:
        df=pd.read_excel(file) if file.name.endswith(".xlsx") else pd.read_csv(file)
        df=normalize_columns(coerce_types(df))
        k,df=calc_kpis(df)
        ch=charts(df)
        summary=ai_summary(build_prompt(k))
        st.markdown(summary)
        pdf_bytes=generate_pdf(summary,k,ch)
        st.download_button("📄 Download Executive Dashboard PDF",pdf_bytes,
                           f"SAP_Report_{datetime.date.today()}.pdf","application/pdf")

elif page=="📬 My Reports Portal":
    st.markdown("### 📦 Access Your Reports")
    email=st.text_input("Registered Email")
    if st.button("Fetch Reports"):
        url=f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}?filterByFormula=FIND('{email}',{{Email}})"
        r=requests.get(url,headers={"Authorization":f"Bearer {AIRTABLE_API_KEY}"})
        if r.status_code==200:
            data=r.json()
            recs=[{"Report Name":x["fields"].get("Report Name"),
                   "Report URL":x["fields"].get("Report URL"),
                   "Created On":x["fields"].get("Created On")} for x in data.get("records",[])]
            if recs:
                for r_ in recs:
                    st.markdown(f"📄 **{r_['Report Name']}** — _{r_['Created On']}_  \n [🔗 Download]({r_['Report URL']})",unsafe_allow_html=True)
            else:st.info("No reports found.")
        else:st.error("Error fetching from Airtable")
