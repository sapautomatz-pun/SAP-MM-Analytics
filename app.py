# ==========================================================
# SAP AUTOMATZ – AI Procurement Analytics (ERP-Compatible)
# Executive Insights Edition (v29.0)
# Hybrid insights: Calculated KPIs + GPT-polished text
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

# -------------------------
# CONFIG
# -------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "https://sapautomatz-backend.onrender.com")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"  # adjust if needed
LOGO_URL = "https://raw.githubusercontent.com/sapautomatz-pun/SAP-MM-Analytics/1d3346d7d35396f13ff06da26f24ebb5ebb70f23/sapautomatz_logo.png"

# Fonts paths (ensure DejaVu exists on host or include fonts folder)
if platform.system() == "Windows":
    FONT_PATH = "./fonts/DejaVuSans.ttf"
    FONT_PATH_BOLD = "./fonts/DejaVuSans-Bold.ttf"
else:
    FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    FONT_PATH_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------
# STREAMLIT PAGE SETUP
# -------------------------
st.set_page_config(page_title="SAP Automatz – AI Procurement Analytics (Executive Insights)", page_icon="📊", layout="wide")
st.markdown("<style>.stApp header{visibility:hidden}</style>", unsafe_allow_html=True)

col1, col2 = st.columns([1,3])
with col1:
    st.image(LOGO_URL, width=150)
with col2:
    st.markdown("""
        <h2 style='margin-bottom:0;color:#1a237e;'>SAP Automatz – AI Procurement Analytics</h2>
        <p style='color:#444;margin-top:0;font-size:14px;'>
        ERP-compatible procurement insights — Hybrid: computed KPIs + GPT-polished analysis.<br>
        <b>Automate. Analyze. Accelerate.</b>
        </p>
    """, unsafe_allow_html=True)
st.divider()

# -------------------------
# HELPERS
# -------------------------
def sanitize_text(text):
    if text is None:
        return ""
    text = unidecode(str(text))
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", "", text)
    return text.strip()

def parse_amount_and_currency(val, fallback_currency=None):
    """
    Attempts to extract numeric amount (float) and currency code from mixed values.
    Handles:
    - "₹ 1,20,000.00", "Rs 1,20,000", "$ 1,234.56", "USD 1234.56"
    - numeric types (1000.5) with optional fallback_currency
    Returns (amount_float, currency_code_or_None)
    """
    if pd.isna(val):
        return 0.0, None
    # If already numeric
    if isinstance(val, (int, float, np.number)):
        return float(val), fallback_currency
    s = str(val).strip()
    # Currency symbols mapping
    sym_map = {"₹":"INR","Rs":"INR","INR":"INR","$":"USD","USD":"USD","€":"EUR","EUR":"EUR","£":"GBP","JPY":"JPY","¥":"JPY"}
    currency = None
    # detect symbol or code
    for sym, code in sym_map.items():
        if s.startswith(sym) or (" " + sym + " ") in (" " + s + " "):
            currency = code
            # remove the symbol/code from string
            s = re.sub(r"^[^\d\-]+","",s)  # strip leading non-numeric
            s = re.sub(r"[A-Za-z]{2,}","",s)  # remove stray letters
            break
    # If currency col present style like "USD 123.45" -> try find code
    m = re.search(r"\b([A-Z]{3})\b", str(val))
    if m:
        currency = m.group(1)
        s = re.sub(r"[A-Z]{3}", "", s)
    # Remove commas and non-digit except dot and minus
    s_clean = re.sub(r"[^\d.\-]", "", s)
    try:
        amt = float(s_clean) if s_clean not in ("", None) else 0.0
    except:
        amt = 0.0
    if currency is None:
        currency = fallback_currency
    return amt, currency

def clean_dataframe_amounts(df):
    """
    Produces two new columns:
    - AMOUNT_NUM: numeric amount converted (for cases where AMOUNT is mixed strings)
    - CURRENCY_DETECTED: best-effort currency code per row (None allowed)
    """
    fallback_currency = None
    if "CURRENCY" in df.columns:
        # normalize currency column values to codes (e.g., INR/USD/EUR)
        def norm_cur(x):
            if pd.isna(x): return None
            x = str(x).strip()
            map_ = {"₹":"INR","Rs":"INR","INR":"INR","USD":"USD","$":"USD","EUR":"EUR","€":"EUR"}
            for k,v in map_.items():
                if k in x:
                    return v
            return x.upper() if len(x)==3 else None
        df["CURRENCY_DETECTED"] = df["CURRENCY"].apply(norm_cur)
        # choose fallback as mode currency if available
        try:
            fallback_currency = df["CURRENCY_DETECTED"].dropna().mode().iloc[0]
        except:
            fallback_currency = None
    else:
        df["CURRENCY_DETECTED"] = None

    # Handle AMOUNT or VALUE columns auto-detect
    amount_col = None
    for c in df.columns:
        if any(k in c.upper() for k in ["AMOUNT","VALUE","TOTAL","PRICE","COST"]):
            amount_col = c
            break
    if amount_col is None:
        # set defaults
        df["AMOUNT_NUM"] = 0.0
        return df

    # Parse each value
    parsed_amounts = []
    parsed_currencies = []
    for idx, row in df.iterrows():
        raw = row[amount_col]
        fallback = row.get("CURRENCY_DETECTED", fallback_currency)
        amt, cur = parse_amount_and_currency(raw, fallback_currency=fallback)
        parsed_amounts.append(amt)
        parsed_currencies.append(cur)
    df["AMOUNT_NUM"] = parsed_amounts
    # if CURRENCY_DETECTED was None, populate from parsed_currencies
    df["CURRENCY_DETECTED"] = df.apply(lambda r: r["CURRENCY_DETECTED"] if r["CURRENCY_DETECTED"] else (r["AMOUNT_NUM"] and parsed_currencies[r.name]), axis=1)
    return df

def compute_kpis(df):
    """
    Compute KPIs needed for the enhanced report:
    - total records
    - totals_by_currency (dict)
    - total_spend_overall (sum numeric amounts)
    - top_vendors_by_spend (series)
    - top_materials_by_quantity (series if QUANTITY exists)
    - monthly_trend (series)
    """
    df = df.copy()
    # make sure dates are parsed
    if "PO_DATE" in df.columns:
        df["PO_DATE_DT"] = pd.to_datetime(df["PO_DATE"], errors="coerce")
    else:
        df["PO_DATE_DT"] = pd.NaT

    # Normalize amounts/currencies
    df = clean_dataframe_amounts(df)
    total_records = len(df)
    # Totals by currency
    totals_by_currency = {}
    if "CURRENCY_DETECTED" in df.columns:
        grouped = df.groupby("CURRENCY_DETECTED")["AMOUNT_NUM"].sum()
        totals_by_currency = {str(k): float(v) for k,v in grouped.to_dict().items() if k and (not math.isnan(v))}
    else:
        totals_by_currency = {"N/A": float(df["AMOUNT_NUM"].sum())}
    total_spend_overall = float(df["AMOUNT_NUM"].sum())

    # Top vendors by spend
    top_vendors = {}
    if "VENDOR" in df.columns:
        try:
            v = df.groupby("VENDOR")["AMOUNT_NUM"].sum().sort_values(ascending=False).head(10)
            top_vendors = v.to_dict()
        except Exception:
            top_vendors = {}
    # Top materials by quantity
    top_materials_by_qty = {}
    if "MATERIAL" in df.columns and "QUANTITY" in df.columns:
        try:
            m = df.groupby("MATERIAL")["QUANTITY"].sum().sort_values(ascending=False).head(10)
            top_materials_by_qty = m.to_dict()
        except Exception:
            top_materials_by_qty = {}
    else:
        # fallback: top materials by spend
        if "MATERIAL" in df.columns:
            try:
                m = df.groupby("MATERIAL")["AMOUNT_NUM"].sum().sort_values(ascending=False).head(10)
                top_materials_by_qty = m.to_dict()
            except:
                top_materials_by_qty = {}

    # Monthly trend (sum by month)
    monthly_trend = {}
    if "PO_DATE_DT" in df.columns:
        try:
            df_dates = df.dropna(subset=["PO_DATE_DT"])
            if not df_dates.empty:
                df_dates["YM"] = df_dates["PO_DATE_DT"].dt.to_period("M").astype(str)
                mt = df_dates.groupby("YM")["AMOUNT_NUM"].sum().sort_index()
                monthly_trend = {k: float(v) for k,v in mt.to_dict().items()}
        except Exception:
            monthly_trend = {}

    # dominant currency
    dominant_currency = None
    if totals_by_currency:
        try:
            dominant_currency = max(totals_by_currency.items(), key=lambda x: x[1])[0]
        except:
            dominant_currency = None

    kpis = {
        "total_records": total_records,
        "totals_by_currency": totals_by_currency,
        "total_spend_overall": total_spend_overall,
        "top_vendors": top_vendors,
        "top_materials_by_qty": top_materials_by_qty,
        "monthly_trend": monthly_trend,
        "dominant_currency": dominant_currency
    }
    return df, kpis

# -------------------------
# HYBRID AI PROMPT (calculated + GPT)
# -------------------------
def generate_hybrid_insights(kpis):
    """
    Build a concise system/user prompt that contains the calculated KPIs
    and ask GPT to produce:
      - Executive Insights (bullet list)
      - Recommendations (bullet list)
      - Key Action Points (bullet list)
    """
    # Prepare short KPI text for prompt
    totals_text = "\n".join([f"- {cur}: {amt:,.2f}" for cur, amt in kpis["totals_by_currency"].items()]) if kpis["totals_by_currency"] else "No currency totals available."
    top_vendors_text = "\n".join([f"{i+1}. {v}: {amt:,.2f}" for i,(v,amt) in enumerate(kpis["top_vendors"].items())]) if kpis["top_vendors"] else "No vendor data."
    top_materials_text = "\n".join([f"{i+1}. {m}: {qty:,.0f}" for i,(m,qty) in enumerate(kpis["top_materials_by_qty"].items())]) if kpis["top_materials_by_qty"] else "No material data."
    monthly_preview = ", ".join(list(kpis["monthly_trend"].keys())[:6]) if kpis["monthly_trend"] else "No monthly data."

    prompt = f"""
You are a concise procurement analyst. Below are computed KPI summaries from a procurement dataset.

Total records: {kpis['total_records']}
Total spend (all currencies): {kpis['total_spend_overall']:,.2f}
Totals by currency:
{totals_text}

Top vendors by purchase amount:
{top_vendors_text}

Top materials (by quantity or spend):
{top_materials_text}

Monthly trend (month keys sample): {monthly_preview}

Task: Using the computed KPIs above, produce 3 sections:
1) Executive Insights — 4 to 6 bullet points (each short, impactful).
2) Recommendations — 4 short bullets focused on procurement optimization.
3) Key Action Points — 5 concise, prioritized action items.

Tone: professional, concise, actionable. Avoid speculation. If data is missing for a section, note it briefly.
Return the three sections separated by clear headings: "Executive Insights:", "Recommendations:", "Key Action Points:".
"""
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role":"system","content":"You are an expert procurement analyst producing executive summaries."},
                {"role":"user","content":prompt}
            ],
            temperature=0.2,
            max_tokens=420
        )
        text = r.choices[0].message.content
        return sanitize_text(text)
    except Exception as e:
        return f"AI Error: {e}"

# -------------------------
# PDF (formatted) Generator
# -------------------------
class PDF(FPDF):
    def header(self): pass
    def footer(self):
        self.set_y(-15)
        try:
            self.set_font("DejaVu","",9)
        except:
            self.set_font("Helvetica","",9)
        self.set_text_color(130,130,130)
        self.cell(0,10,"© 2025 SAP Automatz – AI Procurement Analytics (ERP-Compatible)",align="C")

def add_cover(pdf, customer, key):
    pdf.add_page()
    pdf.set_fill_color(26,35,126)
    pdf.rect(0,0,210,297,'F')
    try:
        pdf.image(LOGO_URL,70,30,70)
    except:
        pass
    pdf.set_text_color(255,255,255)
    pdf.set_font("DejaVu","B",20)
    pdf.ln(110)
    pdf.cell(0,10,"AI Procurement Analytics Report",align="C",ln=True)
    pdf.ln(6)
    pdf.set_font("DejaVu","",12)
    pdf.cell(0,8,f"Customer: {customer}",align="C",ln=True)
    pdf.cell(0,8,f"License Key: {key}",align="C",ln=True)
    pdf.cell(0,8,f"Date: {datetime.date.today().strftime('%d %b %Y')}",align="C",ln=True)

def generate_pdf(ai_text, kpis, charts, customer, key):
    pdf = PDF()
    # fonts registration (use DejaVu for unicode)
    try:
        pdf.add_font("DejaVu","",FONT_PATH,uni=True)
        pdf.add_font("DejaVu","B",FONT_PATH_BOLD,uni=True)
        base_font = "DejaVu"
    except:
        base_font = "Helvetica"

    add_cover(pdf, customer, key)

    # Executive Insights page
    pdf.add_page()
    pdf.set_text_color(0,0,0)
    pdf.set_font(base_font,"B",14)
    pdf.cell(0,10,"Executive Insights",ln=True)
    pdf.ln(4)
    pdf.set_font(base_font,"",11)
    for line in ai_text.split("\n"):
        if line.strip():
            pdf.multi_cell(0,7,line)
    pdf.ln(4)

    # KPI snapshot
    pdf.set_font(base_font,"B",12)
    pdf.cell(0,8,"Summary KPIs",ln=True)
    pdf.set_font(base_font,"",11)
    pdf.multi_cell(0,6,f"Total Records: {kpis['total_records']}")
    pdf.multi_cell(0,6,f"Total Spend (all currencies): {kpis['total_spend_overall']:,.2f}")
    pdf.multi_cell(0,6,f"Dominant Currency: {kpis['dominant_currency']}")
    pdf.ln(3)
    pdf.set_font(base_font,"B",12)
    pdf.cell(0,8,"Totals by Currency",ln=True)
    pdf.set_font(base_font,"",11)
    for cur, amt in kpis['totals_by_currency'].items():
        pdf.multi_cell(0,6,f"{cur}: {amt:,.2f}")

    # KPIs continued - Top vendors and materials
    pdf.ln(4)
    pdf.set_font(base_font,"B",12)
    pdf.cell(0,8,"Top Vendors (by Spend)",ln=True)
    pdf.set_font(base_font,"",11)
    for v,amt in kpis['top_vendors'].items():
        pdf.multi_cell(0,6,f"{v}: {amt:,.2f}")

    pdf.ln(4)
    pdf.set_font(base_font,"B",12)
    pdf.cell(0,8,"Top Materials (by Qty / Spend)",ln=True)
    pdf.set_font(base_font,"",11)
    for m,q in kpis['top_materials_by_qty'].items():
        pdf.multi_cell(0,6,f"{m}: {q:,.0f}")

    # Charts pages
    for path in charts:
        if path and os.path.exists(path):
            pdf.add_page()
            pdf.set_font(base_font,"B",12)
            pdf.cell(0,8,os.path.basename(path).replace("_"," ").title(),ln=True)
            try:
                pdf.image(path, w=160)
            except:
                pdf.multi_cell(0,6,"Chart unavailable.")

    # return bytes stream
    pdf_bytes = pdf.output(dest="S").encode("latin-1", "ignore")
    return io.BytesIO(pdf_bytes)

# -------------------------
# DASHBOARD UI (componentized)
# -------------------------
def show_dashboard():
    st.title("📊 Executive Procurement Dashboard")
    st.write("Upload your SAP/ERP procurement extract (CSV/XLSX). The app computes KPIs and generates GPT-polished insights and recommendations.")

    uploaded = st.file_uploader("Upload Procurement File (CSV/XLSX)", type=["csv","xlsx"], key="file_uploader_main")
    if not uploaded:
        st.info("Use the sample datasets (Clean / Mixed / Missing / Text / Large) to test the report.")
        return

    # Safe read
    try:
        if uploaded.name.lower().endswith(".xlsx"):
            import openpyxl  # ensure available
            df = pd.read_excel(uploaded, engine="openpyxl")
        else:
            df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        return

    # Compute KPIs
    df_processed, kpis = compute_kpis(df)

    # KPI cards row
    c1,c2,c3,c4,c5 = st.columns([1,1,1,1,1])
    c1.metric("Total Records", kpis["total_records"])
    c2.metric("Total Spend (all)", f"{kpis['total_spend_overall']:,.2f} {kpis.get('dominant_currency','')}")
    c3.metric("Dominant Currency", kpis.get("dominant_currency", "N/A"))
    top_vendor = next(iter(kpis['top_vendors']), "N/A")
    c4.metric("Top Vendor", top_vendor)
    top_material = next(iter(kpis['top_materials_by_qty']), "N/A")
    c5.metric("Top Material", top_material)

    st.markdown("### Visual Highlights")
    charts = []

    # Currency distribution pie
    st.subheader("Currency Distribution")
    if kpis['totals_by_currency']:
        cur_labels = list(kpis['totals_by_currency'].keys())
        cur_vals = list(kpis['totals_by_currency'].values())
        fig1, ax1 = plt.subplots(figsize=(5,4))
        ax1.pie(cur_vals, labels=cur_labels, autopct='%1.1f%%', startangle=90)
        ax1.set_title("Currency Distribution")
        cur_chart = "chart_currency.png"
        fig1.tight_layout(); fig1.savefig(cur_chart); charts.append(cur_chart)
        st.pyplot(fig1)
    else:
        st.warning("Not enough currency data to show distribution.")

    # Top vendors bar
    st.subheader("Top 10 Vendors by Purchase Amount")
    if kpis['top_vendors']:
        vendors = list(kpis['top_vendors'].keys())
        vals = list(kpis['top_vendors'].values())
        fig2, ax2 = plt.subplots(figsize=(8,4))
        ax2.barh(vendors[::-1], vals[::-1], color="#2E7D32")
        ax2.set_xlabel("Purchase Amount")
        ax2.set_title("Top Vendors")
        vendor_chart = "chart_vendors.png"
        fig2.tight_layout(); fig2.savefig(vendor_chart); charts.append(vendor_chart)
        st.pyplot(fig2)
    else:
        st.warning("Not enough vendor data to show top vendors.")

    # Top materials by quantity
    st.subheader("Top 10 Materials by Quantity (or spend fallback)")
    if kpis['top_materials_by_qty']:
        mats = list(kpis['top_materials_by_qty'].keys())
        mvals = list(kpis['top_materials_by_qty'].values())
        fig3, ax3 = plt.subplots(figsize=(8,4))
        ax3.bar(mats[::-1], mvals[::-1], color="#1565C0")
        ax3.set_ylabel("Qty / Spend")
        ax3.set_title("Top Materials")
        mat_chart = "chart_materials.png"
        fig3.tight_layout(); fig3.savefig(mat_chart); charts.append(mat_chart)
        st.pyplot(fig3)
    else:
        st.warning("Not enough material data to show top materials.")

    # Monthly trend
    st.subheader("Monthly Purchase Trend")
    if kpis['monthly_trend']:
        months = list(kpis['monthly_trend'].keys())
        vals = list(kpis['monthly_trend'].values())
        fig4, ax4 = plt.subplots(figsize=(9,3))
        ax4.plot(months, vals, marker='o')
        ax4.set_xticklabels(months, rotation=45)
        ax4.set_title("Monthly Purchase Trend")
        ax4.set_ylabel("Spend")
        trend_chart = "chart_monthly.png"
        fig4.tight_layout(); fig4.savefig(trend_chart); charts.append(trend_chart)
        st.pyplot(fig4)
    else:
        st.warning("Not enough dated records to show monthly trend.")

    # Generate hybrid AI insights (calculated KPIs provided)
    st.subheader("Executive Insights & Recommendations")
    with st.spinner("Generating AI-polished insights..."):
        ai_text = generate_hybrid_insights(kpis)
    st.markdown(ai_text.replace("\n", "  \n"))  # preserve newlines as markdown breaks

    # Download PDF
    st.markdown("---")
    pdf_bytes = generate_pdf(ai_text, kpis, charts, "Demo Customer", st.session_state.get("access_key","TEST-KEY"))
    st.download_button("📄 Download Full Executive PDF Report", pdf_bytes, f"SAP_Automatz_Report_{datetime.date.today()}.pdf", "application/pdf")

# -------------------------
# ACCESS CONTROL (minimal gating)
# -------------------------
if "verified" not in st.session_state:
    st.session_state.verified = False

if not st.session_state.verified:
    st.markdown("### 🔐 Verify Access")
    access_key = st.text_input("Enter your access key", type="password", key="access_key_input")
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
else:
    show_dashboard()
