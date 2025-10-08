import streamlit as st
import pandas as pd
import requests
import io

# -------------------- APP SETTINGS --------------------
st.set_page_config(
    page_title="SAP Automatz – Procurement Analytics Auto Generator",
    page_icon="📊",
    layout="centered"
)

# Hide Streamlit default elements
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.title("📊 SAP Automatz – Procurement Analytics Auto Generator")
st.write("Upload your SAP **Purchase Order (PO)** or **GRN** data file (CSV/XLSX). "
         "Our AI system will analyze your procurement performance and email you a summary report.")

# -------------------- FILE UPLOAD --------------------
uploaded_file = st.file_uploader("Upload your SAP PO or GRN file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Read file content into pandas
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ File uploaded successfully!")
        st.dataframe(df.head(5))

        # Convert to CSV in memory
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # Optionally store the file in Airtable / cloud or send to Make.com
        st.info("🔄 Sending file to Make.com workflow for analysis...")

        # Your Make.com webhook URL
        MAKE_WEBHOOK_URL = "https://hook.us1.make.com/YOUR_WEBHOOK_ID"  # <-- Replace with actual

        # Send file to Make.com webhook
        response = requests.post(
            MAKE_WEBHOOK_URL,
            files={"file": (uploaded_file.name, csv_buffer, "text/csv")},
            data={"source": "streamlit_app"}
        )

        if response.status_code == 200:
            st.success("🎉 File sent successfully! You will receive your AI-generated report shortly via email.")
        else:
            st.error(f"⚠️ Failed to send file to Make.com. Response code: {response.status_code}")

    except Exception as e:
        st.error(f"Error processing the file: {e}")

else:
    st.info("📥 Please upload a CSV or Excel file above to begin.")

# -------------------- MANUAL API INFO --------------------
st.markdown("""
---
### 🔗 Manual Upload (Alternative)
If upload fails, you can manually send your file to the Make.com webhook:
