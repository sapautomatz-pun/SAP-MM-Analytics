# ==========================================================
# SAP AUTOMATZ – AI Procurement Analytics (ERP-Compatible)
# Executive Dashboard Edition (v31.0 – Watermark Edition)
# ==========================================================

import os, io, re, datetime, platform, requests, math
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from openai import OpenAI
from fpdf import FPDF
from unidecode import unidecode

# ------------------------- CONFIG -------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "https://sapautomatz-backend.onrender.com")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"
LOGO_URL = "https://raw.githubusercontent.com/sapautomatz-pun/SAP-MM-Analytics/1d3346d7d35396f13ff06da26f24ebb5ebb70f23/sapautomatz_logo.png"

client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------- STREAMLIT -------------------------
st.set_page_config(page_title="SAP Automatz – Executive Procurement Analytics", page_icon="📊", layout="wide")
st.markdown("<style>.stApp header{visibility:hidden}</style>", unsafe_allow_html=True)

col1, col2 = st.columns([1,3])
with col1:
    st.image(LOGO_URL, width=140)
with col2:
    st.markdown("""
        <h2 style='margin-bottom:0;color:#1a237e;font-size:26px;'>SAP Automatz – AI Procurement Analytics</h2>
        <p style='color:#444;margin-top:0;font-size:14px;'>ERP-Compatible Executive Dashboard<br>
        <b>Automate. Analyze. Accelerate.</b></p>
    """, unsafe_allow_html=True)
st.divider()

# ------------------------- HELPERS -------------------------
def sanitize_text(t):
    return re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", "", unidecode(str(t))) if t is not None else ""

def parse_amount_and_currency(v, fallback="INR"):
    if pd.isna(v): return 0.0, fallback
    if isinstance(v,(int,float,np.number)): return float(v), fallback
    s=str(v); sym_map={"₹":"INR","Rs":"INR","$":"USD","USD":"USD","€":"EUR","EUR":"EUR"}
    cur=fallback
    for sym,c in sym_map.items():
        if sym in s: cur=c; s=s.replace(sym,"")
    s=re.sub(r"[^\d.\-]","",s)
    try: amt=float(s)
    except: amt=0.0
    return amt,cur

def clean_dataframe(df):
    if "CURRENCY" not in df.columns: df["CURRENCY"]="INR"
    amt=[];cur=[]
    for _,r in df.iterrows():
        a,c=parse_amount_and_currency(r.get("AMOUNT",0),r.get("CURRENCY","INR"))
        amt.append(a);cur.append(c)
    df["AMOUNT_NUM"]=amt;df["CURRENCY_DETECTED"]=cur
    return df

def compute_kpis(df):
    df=clean_dataframe(df)
    if "PO_DATE" in df.columns:
        df["PO_DATE"]=pd.to_datetime(df["PO_DATE"],errors="coerce")

    totals=df.groupby("CURRENCY_DETECTED")["AMOUNT_NUM"].sum().to_dict()
    total_spend=sum(totals.values())
    dominant=max(totals,key=totals.get)
    top_v=df.groupby("VENDOR")["AMOUNT_NUM"].sum().nlargest(10).to_dict() if "VENDOR" in df.columns else {}
    top_m=df.groupby("MATERIAL")["QUANTITY"].sum().nlargest(10).to_dict() if "QUANTITY" in df.columns else {}
    monthly={}
    if "PO_DATE" in df.columns:
        d=df.dropna(subset=["PO_DATE"])
        if not d.empty:
            d["YM"]=d["PO_DATE"].dt.to_period("M").astype(str)
            monthly=d.groupby("YM")["AMOUNT_NUM"].sum().to_dict()
    return {"totals":totals,"total_spend":total_spend,"dominant":dominant,
            "top_v":top_v,"top_m":top_m,"monthly":monthly,"records":len(df)}

def generate_ai(k):
    prompt=f"""Provide concise professional procurement insights based on:
Totals {k['totals']}, top vendors {list(k['top_v'].keys())[:5]}, top materials {list(k['top_m'].keys())[:5]}, months {list(k['monthly'].keys())[:6]}.
Return three sections titled Executive Insights, Recommendations, and Key Action Points (bulleted)."""
    try:
        r=client.chat.completions.create(model=MODEL,messages=[{"role":"user","content":prompt}],temperature=0.3,max_tokens=400)
        return sanitize_text(r.choices[0].message.content)
    except Exception as e: return f"AI Error: {e}"

# ------------------------- PDF WITH WATERMARK -------------------------
class PDF(FPDF):
    def header(self): pass
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica","I",8)
        self.set_text_color(130,130,130)
        self.cell(0,10,"© 2025 SAP Automatz – Executive Procurement Analytics",align="C")

    def watermark(self):
        self.set_text_color(200,200,200)
        self.set_font("Helvetica","B",50)
        self.rotate(45)
        self.text(30,150,"SAP Automatz")
        self.rotate(0)

    def rotate(self,angle,x=None,y=None):
        if angle!=0:
            self._out(f"q {math.cos(angle*math.pi/180):.5f} {math.sin(angle*math.pi/180):.5f} {-math.sin(angle*math.pi/180):.5f} {math.cos(angle*math.pi/180):.5f} 0 0 cm")
        else:
            self._out("Q")

def add_tile(pdf,x,y,w,h,title,value,color):
    pdf.set_fill_color(*color);pdf.rect(x,y,w,h,"F")
    pdf.set_text_color(255,255,255)
    pdf.set_xy(x+3,y+3);pdf.set_font("Helvetica","B",11);pdf.cell(w-6,7,title,ln=True)
    pdf.set_xy(x+3,y+11);pdf.set_font("Helvetica","",10);pdf.cell(w-6,6,str(value),ln=True)

def generate_pdf(ai,k,charts):
    pdf=PDF()
    pdf.add_page()
    pdf.watermark()

    pdf.set_fill_color(26,35,126);pdf.rect(0,0,210,25,"F")
    pdf.set_text_color(255,255,255);pdf.set_font("Helvetica","B",16)
    pdf.cell(0,15,"Executive Procurement Report",align="C",ln=True)
    pdf.ln(12);pdf.set_text_color(0,0,0)

    y=pdf.get_y()
    add_tile(pdf,10,y,60,22,"Total Records",k["records"],(13,71,161))
    add_tile(pdf,75,y,60,22,"Total Spend",f"{k['total_spend']:,.2f}",(21,101,192))
    add_tile(pdf,140,y,60,22,"Dominant Currency",k["dominant"],(30,136,229))
    pdf.ln(35)

    pdf.set_font("Helvetica","B",13);pdf.cell(0,10,"Executive Insights & Recommendations",ln=True)
    pdf.set_font("Helvetica","",11)
    for line in ai.split("\n"):
        if line.strip(): pdf.multi_cell(0,7,line.strip())

    for ch in charts:
        if os.path.exists(ch):
            pdf.add_page();pdf.watermark()
            pdf.set_font("Helvetica","B",12)
            pdf.cell(0,10,os.path.basename(ch).replace("_"," ").title(),ln=True)
            pdf.image(ch,x=20,w=170)
    return io.BytesIO(pdf.output(dest="S").encode("latin-1"))

# ------------------------- STREAMLIT UI -------------------------
st.title("📊 Executive Procurement Dashboard")
st.caption("Upload your SAP/ERP extract to view insights and export branded PDF report.")

f=st.file_uploader("Upload CSV or XLSX",type=["csv","xlsx"])
if not f: st.stop()
df=pd.read_excel(f) if f.name.endswith(".xlsx") else pd.read_csv(f)
k=compute_kpis(df)

# Adjust metric font size
st.markdown("<style>.stMetric label{font-size:12px!important}</style>",unsafe_allow_html=True)
c1,c2,c3,c4=st.columns(4)
c1.metric("Total Records",k["records"])
c2.metric("Total Spend",f"{k['total_spend']:,.2f} {k['dominant']}")
c3.metric("Top Vendor",next(iter(k["top_v"]), "N/A"))
c4.metric("Top Material",next(iter(k["top_m"]), "N/A"))

charts=[]

# Currency pie
st.subheader("Currency Distribution")
fig1,ax1=plt.subplots()
ax1.pie(list(k["totals"].values()),labels=list(k["totals"].keys()),autopct="%1.1f%%")
fig1.tight_layout();fig1.savefig("chart_currency.png");charts.append("chart_currency.png");st.pyplot(fig1)

# Vendors
st.subheader("Top 10 Vendors by Purchase Amount")
fig2,ax2=plt.subplots()
v=list(k["top_v"].keys());vals=list(k["top_v"].values())
ax2.barh(v[::-1],vals[::-1],color="#2E7D32");ax2.set_xlabel("Amount");fig2.tight_layout()
fig2.savefig("chart_vendors.png");charts.append("chart_vendors.png");st.pyplot(fig2)

# Materials
st.subheader("Top 10 Materials by Quantity")
fig3,ax3=plt.subplots()
m=list(k["top_m"].keys());q=list(k["top_m"].values())
ax3.bar(m,q,color="#1565C0");plt.xticks(rotation=45,ha='right')
fig3.tight_layout();fig3.savefig("chart_materials.png");charts.append("chart_materials.png");st.pyplot(fig3)

# Monthly trend
if k["monthly"]:
    st.subheader("Monthly Purchase Trend")
    fig4,ax4=plt.subplots()
    months=list(k["monthly"].keys());vals=list(k["monthly"].values())
    ax4.plot(months,vals,marker="o");ax4.set_ylabel("Spend")
    plt.xticks(rotation=45,ha="right");fig4.tight_layout()
    fig4.savefig("chart_monthly.png");charts.append("chart_monthly.png");st.pyplot(fig4)

st.markdown("### AI Insights")
ai=generate_ai(k)
st.markdown(ai.replace("\n","  \n"))

pdf=generate_pdf(ai,k,charts)
st.download_button("📄 Download Watermarked PDF Report",pdf,"SAP_Automatz_Executive_Report.pdf","application/pdf")
