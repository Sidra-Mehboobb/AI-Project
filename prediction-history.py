import streamlit as st
import pandas as pd
from datetime import datetime
import os
import time

# Set page config
st.set_page_config(
    page_title="Disease Prediction History",
    layout="wide",
    page_icon="📊"
)

# Custom CSS for background and header
st.markdown(
    """
    <style>
      .stApp {
        background-image: url("https://www.boffinaccess.com/public/frontend/recently-published-articles/a-case-of-post-COVID-19.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        height: 100vh;
      }
      .main::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(255, 255, 255, 0.85);
        z-index: 1;
      }
      .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 0.5rem;
        max-width: 95%;
        position: relative;
        z-index: 2;
      }
      .stButton > button {
        background-color: #000080;
        color: #FFFFFF;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        margin: 5px;
      }
      .stButton > button:hover {
        background-color: #0000A0;
        color: #FFFFFF;
        cursor: pointer;
      }
      .stMarkdown p {
        font-size: 18px;
        color: #333;
      }
       .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #ADD8E6;
        padding: 10px;
        border-radius: 5px;
        margin-top: 0.2rem;
        position: relative;
      }
      .header-title {
        color: #FFFFFF;
        font-size: 16px;
        margin: 0;
      }
      .header-right {
        display: flex;
        align-items: center;
        gap: 10px;
      }
      .profile-icon {
        font-size: 24px;
        color: #00008B;
        margin-right: 10px;
        background-color: #FFFFFF;
        border-radius: 50%;
        padding: 5px;
        cursor: pointer;
      }
      .about-button {
        background-color: #000080;
        color: #FFFFFF;
        border-radius: 5px;
        padding: 2px 7px;
        font-size: 24px;
        border: none;
        cursor: pointer;
      }
      .about-button:hover {
        background-color: #0000A0;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# Header with profile icon and About icon button
st.markdown(
    """
    <div class="header-container">
        <h1 class="header-title">
            <span style="color: #00008B;">🩺</span> IntelliHealth: AI-Based Disease Predictor
        </h1>
        <div class="header-right">
            <button class="about-button" onclick="document.getElementById('about_button').click()">ℹ️</button>
            <span class="profile-icon">👤</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# File path (same directory as script)
DATA_FILE = "predictions.csv"

# --- Initialize Data File (if missing) ---
def init_data_file():
    """Create empty CSV if it doesn't exist."""
    empty_data = {
        "Prediction ID": [],
        "Date": [],
        "Patient ID": [],
        "Disease": [],
        "Prediction": []
    }
    pd.DataFrame(empty_data).to_csv(DATA_FILE, index=False)
    st.success(f"✅ Created empty {DATA_FILE}. Your ML model can now append predictions.")

# --- Load Data (with error handling) ---
@st.cache_data(ttl=5)  # Refresh every 5 seconds
def load_data():
    try:
        # If file doesn't exist, create it
        if not os.path.exists(DATA_FILE):
            init_data_file()
            return pd.DataFrame(columns=["Prediction ID", "Date", "Patient ID", "Disease", "Prediction"])
        
        # Read CSV
        df = pd.read_csv(DATA_FILE)
        
        # Validate expected columns
        expected_columns = ["Prediction ID", "Date", "Patient ID", "Disease", "Prediction"]
        if not all(col in df.columns for col in expected_columns):
            st.warning(f"{DATA_FILE} has unexpected columns. Expected: {expected_columns}")
            df = df[expected_columns] if set(expected_columns).issubset(df.columns) else pd.DataFrame(columns=expected_columns)
        
        # Convert 'Date' to datetime
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        
        return df
    
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return pd.DataFrame(columns=["Prediction ID", "Date", "Patient ID", "Disease", "Prediction"])

# --- Auto-Refresh Logic ---
def check_for_updates():
    """Check if file was modified and trigger rerun."""
    if os.path.exists(DATA_FILE):
        last_modified = os.path.getmtime(DATA_FILE)
        if "last_modified" not in st.session_state:
            st.session_state.last_modified = last_modified
        elif last_modified > st.session_state.last_modified:
            st.session_state.last_modified = last_modified
            st.rerun()

# --- Main App ---
df = load_data()

# Auto-refresh every 3 seconds
with st.empty():
    check_for_updates()
    time.sleep(3)

# --- Filters ---
st.subheader("🔍 Filters")
col1, _ = st.columns(2)
with col1:
    diseases = ["Hypertension", "Heart Disease", "Diabetes"]
    selected_diseases = st.multiselect(
        "Select Diseases",
        options=diseases,
        default=diseases
    )

# Filter data
filtered_df = df[df["Disease"].isin(selected_diseases)]

# --- Display Data ---
st.subheader("📜 Prediction Records")
if not filtered_df.empty:
    st.dataframe(
        filtered_df,
        column_config={
            "Date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD HH:mm"),
            "Prediction ID": st.column_config.TextColumn("Prediction ID"),
            "Patient ID": st.column_config.TextColumn("Patient ID"),
            "Disease": st.column_config.TextColumn("Disease"),
            "Prediction": st.column_config.TextColumn("Prediction")
        },
        hide_index=True,
        use_container_width=True
    )
else:
    st.warning("No predictions found. Add data to 'predictions.csv' via your ML model.")

# --- Export Button ---
if st.button("💾 Export to CSV"):
    export_file = "prediction_history_export.csv"
    filtered_df.to_csv(export_file, index=False)
    st.success(f"Exported to **{export_file}**")

# --- Footer ---
st.markdown("---")
st.caption(f"🔄 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")