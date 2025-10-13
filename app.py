# ==========================================================
# SAP AUTOMATZ – Executive Procurement Analytics
# Version: v33.2 (Fixed Watermark Visibility + Risk Chart Label)
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

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------- STREAMLIT PAGE ----------------
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

# ---------------- HELPERS ----------------
def sanitize_text(t):
    return unidecode(str(t)) if t else ""

def parse_amount_and_currency(v, fallback="INR"):
    if pd.isna(v): return 0.0, fallback
    if isinstance(v,(int,float,np.number)): return float(v), fallback
    s=str(v)
    sym_map={"₹":"INR","Rs":"INR","$":"USD","USD":"USD","€":"EUR","EUR":"EUR"}
    cur=fallback
    for sym,c in sym_map.items():
        if sym in s:
            cur=c
            s=s.replace(sym,"")
    s=re.sub(r"[^\d.\-]", "", s)
    try: amt=float(s)
    except: amt=0.0
    return amt, cur

def clean_dataframe(df):
    if "CURRENCY" not in df.columns: df["CURRENCY"]="INR"
    amt=[];cur=[]
    for _,r in df.iterrows():
        a,c=parse_amount_and_currency(r.get("AMOUNT",0),r.get("CURRENCY","INR"))
        amt.append(a);cur.append(c)
    df["AMOUNT_NUM"]=amt;df["CURRENCY_DETECTED"]=cur
    if "VENDOR" in df.columns: df["VENDOR"]=df["VENDOR"].astype(str).fillna("Unknown")
    if "MATERIAL" in df.columns: df["MATERIAL"]=df["MATERIAL"].astype(str).fillna("Unknown")
    return df

def compute_kpis(df):
    df=clean_dataframe(df)
    if "PO_DATE" in df.columns:
        df["PO_DATE"]=pd.to_datetime(df["PO_DATE"], errors="coerce")
    totals=df.groupby("CURRENCY_DETECTED")["AMOUNT_NUM"].sum().to_dict()
    total_spend=sum(totals.values()) if totals else 0.0
    dominant=max(totals,key=totals.get) if totals else None
    top_v=df.groupby("VENDOR")["AMOUNT_NUM"].sum().nlargest(10).to_dict() if "VENDOR" in df.columns else {}
    if "QUANTITY" in df.columns:
        top_m=df.groupby("MATERIAL")["QUANTITY"].sum().nlargest(10).to_dict()
    else:
        top_m=df.groupby("MATERIAL")["AMOUNT_NUM"].sum().nlargest(10).to_dict() if "MATERIAL" in df.columns else {}
    monthly={}
    if "PO_DATE" in df.columns:
        d=df.dropna(subset=["PO_DATE"])
        if not d.empty:
            d["YM"]=d["PO_DATE"].dt.to_period("M").astype(str)
            monthly=d.groupby("YM")["AMOUNT_NUM"].sum().to_dict()
    return {"totals":totals,"total_spend":total_spend,"dominant":dominant,
            "top_v":top_v,"top_m":top_m,"monthly":monthly,"records":len(df),"df":df}

# ---------------- RISK ----------------
def compute_procurement_risk(df,k):
    df_local=k.get("df",df)
    totals=k.get("totals",{})
    total_spend=k.get("total_spend",0.0)
    v=df_local.groupby("VENDOR")["AMOUNT_NUM"].sum()
    nv=v.size if not v.empty else 0
    top_share=(v.max()/total_spend) if total_spend and not v.empty else 1.0
    v_conc=max(0.0,(1.0-top_share))*100
    v_div=min(100.0,(nv/50)*100)
    if totals and total_spend:
        dom=k.get("dominant")
        dom_share=totals.get(dom,0.0)/total_spend if dom else 1.0
        c_expo=dom_share*100
    else: c_expo=100.0
    mvals=list(k.get("monthly",{}).values())
    if len(mvals)>=3 and np.mean(mvals)>0:
        cv=np.std(mvals)/(np.mean(mvals)+1e-9)
        m_vol=max(0.0,1-min(cv,2))*100
    else: m_vol=80.0
    w_conc=w_div=w_curr=w_vol=0.25
    score=v_conc*w_conc+v_div*w_div+c_expo*w_curr+m_vol*w_vol
    score=float(max(0.0,min(100.0,score)))
    band="Low" if score>=67 else ("Medium" if score>=34 else "High")
    return {"score":score,"band":band,"breakdown":{"Vendor Concentration":v_conc,
            "Vendor Diversity":v_div,"Currency Exposure":c_expo,"Monthly Volatility":m_vol}}

# ---------------- AI ----------------
def generate_ai(k):
    t="\n".join([f"{c}: {v:,.2f}" for c,v in k["totals"].items()])
    v="\n".join([f"{i+1}. {x}: {y:,.2f}" for i,(x,y) in enumerate(k["top_v"].items())])
    prompt=f"""Provide concise executive insights, recommendations, and key actions
for this procurement dataset.

Total spend: {k['total_spend']:,.2f}
Totals:
{t}
Top vendors:
{v}
"""
    try:
        r=client.chat.completions.create(model=MODEL,
            messages=[{"role":"system","content":"You are a procurement analytics expert."},
                      {"role":"user","content":prompt}],
            temperature=0.2,max_tokens=900)
        return sanitize_text(r.choices[0].message.content)
    except Exception as e:
        return f"AI Error: {e}"

# ---------------- PDF CLASS ----------------
class PDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica","I",8)
        self.set_text_color(130,130,130)
        self.cell(0,10,"© 2025 SAP Automatz – Executive Procurement Analytics",align="C")
    def add_watermark(self,text="SAP Automatz – Automate • Analyze • Accelerate"):
        self.set_text_color(250,250,250)
        self.set_font("Helvetica","B",28)
        self.rotate(45)
        self.text(25,150,text)
        self.rotate(0)
    def rotate(self,angle):
        if angle!=0:
            self._out(f"q {math.cos(angle*math.pi/180):.5f} {math.sin(angle*math.pi/180):.5f} "
                      f"{-math.sin(angle*math.pi/180):.5f} {math.cos(angle*math.pi/180):.5f} 0 0 cm")
        else:self._out("Q")

def add_tile(pdf,x,y,w,h,title,value,color):
    pdf.set_fill_color(*color)
    pdf.rect(x,y,w,h,"F")
    pdf.set_text_color(255,255,255)
    pdf.set_xy(x+3,y+4)
    pdf.set_font("Helvetica","B",11)
    pdf.cell(w-6,6,title,ln=True)
    pdf.set_xy(x+3,y+11)
    pdf.set_font("Helvetica","",10)
    pdf.cell(w-6,6,str(value),ln=True)

def safe_text(pdf,text):
    for line in str(text).split("\n"):
        if not line.strip(): continue
        try: pdf.multi_cell(0,7,sanitize_text(line.strip()))
        except: pdf.multi_cell(0,7,sanitize_text(line.strip())[:200])

def generate_pdf(ai,k,charts,company,summary,risk):
    pdf=PDF()
    pdf.add_page()
    pdf.set_font("Helvetica","B",20)
    pdf.cell(0,10,"Executive Procurement Analysis Report",ln=True,align="C")
    pdf.set_font("Helvetica","",12)
    pdf.cell(0,10,f"Prepared for: {company}",ln=True,align="C")
    pdf.cell(0,10,f"Generated on: {datetime.date.today().strftime('%d %B %Y')}",ln=True,align="C")
    pdf.ln(10); pdf.set_font("Helvetica","",11); safe_text(pdf,summary)
    pdf.add_watermark()

    pdf.add_page()
    pdf.set_font("Helvetica","B",14)
    pdf.cell(0,10,"Executive Dashboard",ln=True,align="C")
    y=pdf.get_y()+5
    add_tile(pdf,10,y,50,22,"Total Records",k["records"],(13,71,161))
    add_tile(pdf,65,y,70,22,"Total Spend",f"{k['total_spend']:,.2f}",(21,101,192))
    add_tile(pdf,140,y,60,22,"Dominant Currency",k["dominant"],(30,136,229))
    pdf.ln(36)
    color=(56,142,60) if risk["band"]=="Low" else ((242,153,74) if risk["band"]=="Medium" else (192,39,0))
    add_tile(pdf,10,pdf.get_y(),80,22,"Procurement Risk Index",f"{risk['score']:.0f} ({risk['band']})",color)
    pdf.add_watermark()

    pdf.add_page()
    pdf.set_font("Helvetica","B",13)
    pdf.cell(0,10,"AI-Generated Insights",ln=True)
    pdf.set_font("Helvetica","",11)
    safe_text(pdf,ai)
    pdf.add_watermark()

    pdf.add_page()
    pdf.set_font("Helvetica","B",13)
    pdf.cell(0,10,"Procurement Risk Breakdown",ln=True)
    pdf.set_font("Helvetica","",11)
    for kx,vx in risk["breakdown"].items():
        pdf.cell(0,8,f"{kx}: {vx:,.2f}",ln=True)
    pdf.add_watermark()

    for ch in charts:
        if os.path.exists(ch):
            pdf.add_page()
            pdf.set_font("Helvetica","B",12)
            pdf.cell(0,10,os.path.basename(ch).replace(".png","").replace("_"," ").title(),ln=True)
            pdf.image(ch,x=20,y=30,w=170)
            pdf.add_watermark()

    return io.BytesIO(pdf.output(dest="S").encode("latin-1","ignore"))

# ---------------- RISK GAUGE ----------------
def plot_risk_gauge(score,path="gauge_risk.png"):
    fig,ax=plt.subplots(figsize=(6,3))
    ax.axis("off")
    angles=np.linspace(-np.pi,0,100)
    colors=[(1,0.2,0.2),(1,0.7,0.2),(0.2,0.7,0.2)]
    splits=[0,33,66,100]
    for i in range(3):
        start=-np.pi+(splits[i]/100)*np.pi;end=-np.pi+(splits[i+1]/100)*np.pi
        t=np.linspace(start,end,50);ax.fill_between(np.cos(t),np.sin(t),-1.2,color=colors[i],alpha=0.9)
    th=-np.pi+(score/100)*np.pi;x=0.9*math.cos(th);y=0.9*math.sin(th)
    ax.plot([0,x],[0,y],lw=4,color="k");ax.scatter([0],[0],color="k",s=30)
    ax.text(0,-0.1,f"{score:.0f}",ha="center",va="center",fontsize=20,fontweight="bold")
    ax.set_xlim(-1.2,1.2);ax.set_ylim(-1.2,0.4)
    fig.savefig(path,bbox_inches="tight",dpi=150);plt.close(fig)
    return path

# ---------------- MAIN UI ----------------
st.title("📊 Executive Procurement Dashboard")
company_name=st.text_input("Enter Company Name:","ABC Manufacturing Pvt Ltd")
f=st.file_uploader("Upload CSV/XLSX",type=["csv","xlsx"])
if not f: st.stop()
df=pd.read_excel(f) if f.name.endswith(".xlsx") else pd.read_csv(f)
k=compute_kpis(df)
risk=compute_procurement_risk(df,k)
gauge=plot_risk_gauge(risk["score"])

charts=[gauge]
if k["totals"]:
    fig,ax=plt.subplots();ax.pie(k["totals"].values(),labels=k["totals"].keys(),autopct="%1.1f%%",startangle=90)
    ax.set_title("Currency Distribution");fig.savefig("chart_currency.png",bbox_inches="tight",dpi=150);plt.close(fig);charts.append("chart_currency.png")
if k["top_v"]:
    fig,ax=plt.subplots();ax.barh(list(k["top_v"].keys())[::-1],list(k["top_v"].values())[::-1],color="#2E7D32")
    ax.set_title("Top Vendors by Spend");fig.savefig("chart_vendors.png",bbox_inches="tight",dpi=150);plt.close(fig);charts.append("chart_vendors.png")
if k["top_m"]:
    fig,ax=plt.subplots();ax.bar(list(k["top_m"].keys()),list(k["top_m"].values()),color="#1565C0");plt.xticks(rotation=45,ha="right")
    ax.set_title("Top Materials by Quantity/Spend");fig.savefig("chart_materials.png",bbox_inches="tight",dpi=150);plt.close(fig);charts.append("chart_materials.png")
if k["monthly"]:
    fig,ax=plt.subplots();ax.plot(list(k["monthly"].keys()),list(k["monthly"].values()),marker="o");plt.xticks(rotation=45,ha="right")
    ax.set_title("Monthly Purchase Trend");fig.savefig("chart_monthly.png",bbox_inches="tight",dpi=150);plt.close(fig);charts.append("chart_monthly.png")

# KPI cards
c1,c2,c3,c4=st.columns(4)
c1.metric("Records",k["records"])
c2.metric("Spend",f"{k['total_spend']:,.2f} {k['dominant']}")
c3.metric("Top Vendor",next(iter(k["top_v"]),"N/A"))
c4.metric("Risk",f"{risk['score']:.0f} ({risk['band']})")

st.subheader("Procurement Risk Index Gauge")
st.image(gauge, use_container_width=True, caption="Procurement Risk Index Gauge")

st.subheader("Procurement Risk Breakdown")
st.table(pd.DataFrame.from_dict(risk["breakdown"],orient="index",columns=["Score"]).reset_index().rename(columns={"index":"Metric"}))

st.subheader("Visual Highlights")
for ch in charts[1:]:
    st.image(ch, use_container_width=True)

st.subheader("AI Insights")
ai=generate_ai(k)
st.markdown(ai.replace("\n","  \n"))
summary=ai[:1000]
pdf=generate_pdf(ai,k,charts,company_name,summary,risk)
st.download_button("📄 Download Full Executive Report",pdf,"SAP_Automatz_Executive_Report.pdf","application/pdf")
