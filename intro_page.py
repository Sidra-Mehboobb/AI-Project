import streamlit as st
import time

# Set page configuration with no sidebar
st.set_page_config(page_title="IntelliHealth", layout="wide", initial_sidebar_state="collapsed")

# Inject CSS for background image, title, and button styling
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
    h1 {
        margin-top: 10px !important;
    }
    .stButton>button {
        background-color: #001f3f;
        color: white;
        font-size: 20px;
        padding: 12px 30px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.3s ease;
        display: block;
        margin: auto;
    }
    .stButton>button:hover {
        background-color: #003366;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title animation logic
if 'title_index' not in st.session_state:
    st.session_state.title_index = 0
    st.session_state.animated_title = ""

full_title = "IntelliHealth: AI-Based Disease Predictor"
if st.session_state.title_index <= len(full_title):
    st.session_state.animated_title = full_title[:st.session_state.title_index]
    st.session_state.title_index += 1
    time.sleep(0.05)
    st.rerun()

# Show animated title
st.markdown(f"""
    <h1 style='text-align:center; color:#007acc; margin-top: 10px;'>{st.session_state.animated_title}</h1>
    """, unsafe_allow_html=True)

# Image slider logic
if 'img_index' not in st.session_state:
    st.session_state.img_index = 0
if 'last_time' not in st.session_state:
    st.session_state.last_time = time.time()

images = ["diabetes.jpeg", "heart_disease.jpeg", "hypertension.jpeg"]

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(images[st.session_state.img_index], width=600)

# Rotate image every 3 seconds
if time.time() - st.session_state.last_time > 3:
    st.session_state.img_index = (st.session_state.img_index + 1) % len(images)
    st.session_state.last_time = time.time()
    st.rerun()

# Functional, styled Streamlit button centered
colA, colB, colC = st.columns([1, 1, 1])
with colB:
    if st.button("Start", key="start_button"):
        st.switch_page("pages/app.py")  # Adjust path if necessary
