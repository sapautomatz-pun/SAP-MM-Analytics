import streamlit as st
import pandas as pd
from io import BytesIO
import base64
import datetime
import os
import json

st.set_page_config(page_title="SAP MM Procurement Analytics", layout="wide")

st.title("📊 SAP MM Procurement Analytics - Auto Report Generator")
st.caption("Upload your SAP Purchase Order or GRN file (.csv / .xlsx) to get a summarized analytics report.")

# ---------- UTILITIES ---------- #

def generate_summary(df):
    df.columns = [c.strip().lower() for c in df.columns]
    summary, vendor_summary, monthly = {}, pd.DataFrame(), pd.DataFrame()

    if 'vendor' in df.columns and 'po value' in df.columns:
        vendor_summary = df.groupby('vendor')['po value'].sum().reset_index()
        summary['total_po_value'] = df['po value'].sum()
        summary['top_vendors'] = vendor_summary.sort_values('po value', ascending=False).head(5)

    if 'posting date' in df.columns:
        df['posting date'] = pd.to_datetime(df['posting date'], errors='coerce')
        monthly = df.groupby(df['posting date'].dt.to_period('M')).size().reset_index(name='count')
        monthly['posting date'] = monthly['posting date'].astype(str)

    return summary, vendor_summary, monthly


def generate_excel_report(vendor_summary, monthly_summary):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        vendor_summary.to_excel(writer, index=False, sheet_name='Top Vendors')
        monthly_summary.to_excel(writer, index=False, sheet_name='Monthly Trend')
    output.seek(0)
    return output


def save_and_get_link(excel_data):
    os.makedirs("reports", exist_ok=True)
    filename = f"Procurement_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    filepath = os.path.join("reports", filename)
    with open(filepath, "wb") as f:
        f.write(excel_data.read())
    return f"https://sapautomatz.streamlit.app/reports/{filename}"


def make_download_link(data, filename):
    b64 = base64.b64encode(data.read()).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">📥 Download {filename}</a>'

# ---------- FILE UPLOAD ---------- #

uploaded_file = st.file_uploader("Upload your SAP PO or GRN file", type=["csv", "xlsx"])
query_params = st.experimental_get_query_params()

# ----- Case 1: Manual upload from browser ----- #
if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
    st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")

    summary, vendor_summary, monthly = generate_summary(df)

    if not vendor_summary.empty:
        st.subheader("📈 Top 5 Vendors by PO Value")
        st.dataframe(vendor_summary)
        st.bar_chart(vendor_summary.set_index('vendor')['po value'])

    if not monthly.empty:
        st.subheader("📅 Monthly PO Count Trend")
        st.line_chart(monthly.set_index('posting date')['count'])

    excel_report = generate_excel_report(vendor_summary, monthly)
    download_link = make_download_link(excel_report, f"Procurement_Report_{datetime.date.today()}.xlsx")
    st.markdown(download_link, unsafe_allow_html=True)

# ----- Case 2: Called from Make.com (HTTP request with ?api=1) ----- #
elif "api" in query_params:
    # Example URL: https://sapautomatz.streamlit.app/?api=1&fileurl=https://example.com/file.csv
    file_url = query_params.get("fileurl", [None])[0]
    if file_url:
        try:
            df = pd.read_csv(file_url)
            _, vendor_summary, monthly = generate_summary(df)
            excel_data = generate_excel_report(vendor_summary, monthly)
            report_link = save_and_get_link(excel_data)
            st.json({"status": "success", "report_link": report_link})
        except Exception as e:
            st.json({"status": "error", "message": str(e)})
    else:
        st.json({"status": "error", "message": "No file URL provided."})
else:
    st.info("Upload file manually above or call this endpoint via Make.com with ?api=1&fileurl=<public_csv_url>")
