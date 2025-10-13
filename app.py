# ==========================================================
# SAP AUTOMATZ – Executive Procurement Analytics
# Version: v38.1 (Font Fix + Signature + Final Polished Report)
# ==========================================================

import os, io, re, datetime, math, urllib.request
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
def sanitize_text(t): 
    return unidecode(str(t)) if t else ""

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
                {"role": "user", "content": f"Provide clear executive insights, key recommendations, and action items for dataset:\n{k}"}
            ],
            temperature=0.2, max_tokens=800
        )
        return sanitize_text(r.choices[0].message.content)
    except Exception as e:
        return f"AI Error: {e}"

# ---------------- GAUGE ----------------
def plot_risk_gauge(score, path="gauge_risk.png"):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.axis("off")
    colors = [(1,0.2,0.2),(1,0.7,0.2),(0.2,0.7,0.2)]
    splits = [0,33,66,100]
    for i in range(3):
        start=-np.pi+(splits[i]/100)*np.pi
        end=-np.pi+(splits[i+1]/100)*np.pi
        t=np.linspace(start,end,50)
        ax.fill_between(np.cos(t),np.sin(t),-1.2,color=colors[i],alpha=0.9)
    th=-np.pi+(score/100)*np.pi
    x=0.9*math.cos(th);y=0.9*math.sin(th)
    ax.plot([0,x],[0,y],lw=4,color="k");ax.scatter([0],[0],color="k",s=30)
    ax.text(0,-0.1,f"{score:.0f}",ha="center",va="center",fontsize=20,fontweight="bold")
    ax.text(0,0.35,"Procurement Risk Index",ha="center",fontsize=12,fontweight="bold",color="#333")
    ax.set_xlim(-1.2,1.2);ax.set_ylim(-1.2,0.5)
    fig.savefig(path,bbox_inches="tight",dpi=150);plt.close(fig)
    return path
# ---------------- PDF CLASS ----------------
class PDF(FPDF):
    def header(self): 
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"SAP Automatz Confidential | Page {self.page_no()} of {{nb}}", 0, 0, "C")

    def rect_tile(self, x, y, w, h, color, title, value):
        self.set_fill_color(*color)
        self.rect(x, y, w, h, "F")
        self.set_xy(x + 3, y + 4)
        self.set_text_color(255, 255, 255)
        self.set_font("DejaVu", "B", 12)
        self.cell(w - 6, 6, title, ln=True)
        self.set_xy(x + 3, y + 12)
        self.set_font("DejaVu", "", 11)
        self.cell(w - 6, 6, str(value))

# ---------------- PDF GENERATION ----------------
def generate_pdf(ai_text, kpis, charts, company, summary_text, risk):
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Font setup (Regular, Bold, Italic)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    italic_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"

    if not os.path.exists(font_path):
        os.makedirs("fonts", exist_ok=True)
        urllib.request.urlretrieve(
            "https://github.com/dejavu-fonts/dejavu-fonts/raw/version_2_37/ttf/DejaVuSans.ttf",
            "fonts/DejaVuSans.ttf"
        )
        font_path = "fonts/DejaVuSans.ttf"

    if not os.path.exists(italic_path):
        urllib.request.urlretrieve(
            "https://github.com/dejavu-fonts/dejavu-fonts/raw/version_2_37/ttf/DejaVuSans-Oblique.ttf",
            "fonts/DejaVuSans-Oblique.ttf"
        )
        italic_path = "fonts/DejaVuSans-Oblique.ttf"

    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.add_font("DejaVu", "B", font_path, uni=True)
    pdf.add_font("DejaVu", "I", italic_path, uni=True)
    pdf.set_font("DejaVu", "", 11)

    # 1️⃣ Cover Page
    pdf.add_page()
    pdf.set_font("DejaVu", "B", 20)
    pdf.cell(0, 15, "Executive Procurement Analysis Report", ln=True, align="C")
    pdf.ln(8)
    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 8, f"Prepared for: {company}", ln=True, align="C")
    pdf.cell(0, 8, f"Generated on: {datetime.date.today().strftime('%d %B %Y')}", ln=True, align="C")
    pdf.ln(15)
    pdf.multi_cell(0, 7, summary_text)
    pdf.image(LOGO_URL, x=160, y=260, w=30)

    # 2️⃣ KPI Summary
    pdf.add_page()
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(0, 10, "Executive Dashboard Overview", ln=True, align="C")
    y = pdf.get_y() + 5
    pdf.rect_tile(10, y, 60, 20, (33, 150, 243), "Total Spend", f"{kpis['total_spend']:,.2f}")
    pdf.rect_tile(75, y, 60, 20, (76, 175, 80), "Top Vendor", next(iter(kpis["top_v"]), "N/A"))
    pdf.rect_tile(140, y, 60, 20, (255, 167, 38), "Currency", kpis.get("dominant", "INR"))
    pdf.rect_tile(10, y + 28, 190, 20, (229, 57, 53), "Risk Index", f"{risk['score']:.0f} ({risk['band']})")

    # 3️⃣ AI Insights
    pdf.add_page()
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(0, 10, "AI-Generated Executive Insights", ln=True)
    pdf.set_font("DejaVu", "", 11)
    pdf.multi_cell(0, 7, ai_text)

    # 4️⃣ Risk Breakdown
    pdf.add_page()
    pdf.set_font("DejaVu", "B", 13)
    pdf.cell(0, 10, "Procurement Risk Breakdown", ln=True)
    pdf.set_font("DejaVu", "", 11)
    for kx, vx in risk["breakdown"].items():
        pdf.cell(0, 8, f"{kx}: {vx:,.2f}", ln=True)

    # 5️⃣ Charts
    for ch in charts:
        if os.path.exists(ch):
            pdf.add_page()
            title = os.path.basename(ch).replace("_", " ").replace(".png", "").title()
            pdf.set_font("DejaVu", "B", 12)
            pdf.cell(0, 10, title, ln=True)
            pdf.image(ch, x=20, y=35, w=170)

    # 6️⃣ Summary of Findings + Signature
    pdf.add_page()
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(0, 10, "Summary of Findings", ln=True)
    pdf.set_font("DejaVu", "", 11)
    pdf.ln(4)
    pdf.multi_cell(
        0, 7,
        f"• Total Spend: {kpis['total_spend']:,.2f} {kpis['dominant']}\n"
        f"• Risk Score: {risk['score']:.0f} ({risk['band']})\n"
        f"• Top Vendor: {next(iter(kpis['top_v']), 'N/A')}\n\n"
        f"Key Recommendations:\n{ai_text[:500]}\n\n"
        "For sustained procurement excellence, focus on vendor diversification, currency risk mitigation, "
        "and inventory optimization.\n\n"
        "_____________________________\n"
        "Prepared by: SAP Automatz AI Suite\n"
        "Empowering Intelligent Procurement Transformation."
    )

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
gauge = plot_risk_gauge(risk["score"])
charts = [gauge]

# ------------- Charts -------------
if k["totals"] and sum(k["totals"].values()) > 0:
    fig, ax = plt.subplots()
    ax.pie(k["totals"].values(), labels=k["totals"].keys(), autopct="%1.1f%%", startangle=90)
    ax.set_title("Currency Distribution")
    fig.savefig("chart_currency.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    charts.append("chart_currency.png")

if k["top_v"]:
    fig, ax = plt.subplots()
    ax.barh(list(k["top_v"].keys())[::-1], list(k["top_v"].values())[::-1], color="#2E7D32")
    ax.set_title("Top Vendors by Spend")
    fig.savefig("chart_vendors.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    charts.append("chart_vendors.png")

if k["top_m"]:
    fig, ax = plt.subplots()
    ax.bar(list(k["top_m"].keys()), list(k["top_m"].values()), color="#1565C0")
    plt.xticks(rotation=45, ha="right")
    ax.set_title("Top Materials by Quantity/Spend")
    fig.savefig("chart_materials.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    charts.append("chart_materials.png")

if k["monthly"] and sum(k["monthly"].values()) > 0:
    fig, ax = plt.subplots()
    ax.plot(list(k["monthly"].keys()), list(k["monthly"].values()), marker="o")
    plt.xticks(rotation=45, ha="right")
    ax.set_title("Monthly Purchase Trend")
    fig.savefig("chart_monthly.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    charts.append("chart_monthly.png")

# ------------- Dashboard -------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", k["records"])
c2.metric("Spend", f"{k['total_spend']:,.2f} {k['dominant']}")
c3.metric("Top Vendor", next(iter(k["top_v"]), "N/A"))
c4.metric("Risk", f"{risk['score']:.0f} ({risk['band']})")

st.subheader("Procurement Risk Gauge")
st.image(gauge, use_container_width=True)

st.subheader("Procurement Risk Breakdown")
st.table(pd.DataFrame.from_dict(risk["breakdown"], orient="index", columns=["Score"])
         .reset_index().rename(columns={"index": "Metric"}))

ai = generate_ai(k)
st.subheader("AI Insights")
st.markdown(ai.replace("\n", "  \n"))

summary = ai[:1000]
pdf = generate_pdf(ai, k, charts, company, summary, risk)
st.download_button("📄 Download Full Executive Report", pdf,
                   "SAP_Automatz_Executive_Report.pdf", "application/pdf")
