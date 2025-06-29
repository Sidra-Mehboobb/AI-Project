import streamlit as st

# Page config
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-image: url('https://t3.ftcdn.net/jpg/03/29/30/64/360_F_329306430_6HfoeFHi7RrazRxXmHoz6sMX5UvHPmCL.jpg');
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
    }

    .header-container {
        text-align: center;
        background-color: #5DADE2;
        padding: 20px 20px;
        border-radius: 0 0 20px 20px;
        color: white;
        animation: fadeIn 1s ease-in-out;
    }

    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 0.5rem;
        max-width: 95%;
        position: relative;
        z-index: 2;
    }

    .about-row {
        display: flex;
        justify-content: center;
        gap: 30px;
        flex-wrap: wrap;
        margin: 40px auto;
        max-width: 1000px;
    }

    .card {
        background-color: white;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
        flex: 1 1 45%;
        min-width: 300px;
        animation: fadeIn 1.2s ease-in-out;
    }

    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    h1, h2 {
        color: #333;
    }

    p {
        font-size: 18px;
        line-height: 1.6;
        color: #333;
    }

    .footer {
        text-align: center;
        margin-top: 50px;
        color: white;
        background-color: rgba(25, 118, 210, 0.9);
        padding: 20px;
        border-radius: 0 0 10px 10px;
    }

    .disease-img-container img {
        border-radius: 10px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .disease-img-container img:hover {
        transform: scale(1.05) translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        cursor: pointer;
    }

    .social-links {
        text-align: center;
        margin-top: 40px;
    }

    .social-button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        padding: 12px 24px;
        margin: 10px;
        border-radius: 30px;
        font-size: 16px;
        color: white !important;
        text-decoration: none !important;
        transition: background-color 0.3s ease, transform 0.2s ease;
    }

    .social-button.github {
        background-color: #333;
    }

    .social-button.linkedin {
        background-color: #0077b5;
    }

    .social-button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
    }

    .social-button i {
        color: white !important;
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
        color: #000000;
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

    <!-- Font Awesome CDN for icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
""", unsafe_allow_html=True)

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


# Mission & Why cards
st.markdown("""
<div class="about-row">
    <div class="card">
        <h2>Our Mission</h2>
        <p>
            We aim to empower individuals and healthcare providers by providing an accurate and easy-to-use platform
            for early detection of chronic diseases. Our system uses intelligent algorithms to analyze medical
            indicators and deliver fast, insightful predictions.
        </p>
    </div>
    <div class="card">
        <h2>Why We Built This</h2>
        <p>
            Chronic diseases like diabetes, heart conditions, and kidney problems affect millions globally.
            Early detection can dramatically improve quality of life and reduce treatment costs. Our web app
            helps users assess their health risks and encourages timely medical consultation.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Diseases Title
st.markdown("""<h2 style="text-align:center; margin-top: 50px; color: white;">Common Diseases We Detect</h2>""", unsafe_allow_html=True)

# Disease Images
col1, col2, col3 = st.columns(3)
image_width = 260

with col1:
    st.markdown('<div class="disease-img-container">', unsafe_allow_html=True)
    st.image("diabetes.jpeg", width=image_width, caption="Diabetes")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="disease-img-container">', unsafe_allow_html=True)
    st.image("heart_disease.jpeg", width=image_width, caption="Heart Disease")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="disease-img-container">', unsafe_allow_html=True)
    st.image("hypertension.jpeg", width=image_width, caption="Cancer Disease")
    st.markdown('</div>', unsafe_allow_html=True)

# Social Media Buttons with White Text & Icons
st.markdown("""
<div class="social-links">
    <a href="https://github.com/" target="_blank" class="social-button github">
        <i class="fab fa-github"></i> GitHub
    </a>
    <a href="https://www.linkedin.com/" target="_blank" class="social-button linkedin">
        <i class="fab fa-linkedin-in"></i> LinkedIn
    </a>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown(
    '<div class="footer">'
    '&copy; 2025 Chronic Disease Detector. All rights reserved.'
    '</div>',
    unsafe_allow_html=True
)
