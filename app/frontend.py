"""
Streamlit Frontend for ExplainMyXray.
Simple UI to upload X-ray images and display explanations.
"""
import streamlit as st
import requests
from PIL import Image
import io

# Configuration
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="ExplainMyXray",
    page_icon="🩻",
    layout="centered",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #E3F2FD;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #999;
        text-align: center;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🩻 ExplainMyXray</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Upload your chest X-ray and get a simple, patient-friendly explanation</p>',
    unsafe_allow_html=True
)

# Check API health
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.json().get("model_ready", False)
    except:
        return False

# File uploader
uploaded_file = st.file_uploader(
    "Choose a Chest X-ray image",
    type=["png", "jpg", "jpeg"],
    help="Upload a frontal chest X-ray image (PA or AP view)"
)

# Display uploaded image
if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Your X-ray")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("📋 Explanation")
        
        if st.button("🔍 Analyze X-ray", type="primary", use_container_width=True):
            with st.spinner("Analyzing your X-ray..."):
                try:
                    # Reset file position
                    uploaded_file.seek(0)
                    
                    # Send to API
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    response = requests.post(
                        f"{API_URL}/explain",
                        files=files,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        explanation = result.get("explanation", "No explanation available")
                        status = result.get("status", "success")
                        
                        if status == "demo":
                            st.warning("⚠️ Running in demo mode (model not loaded)")
                        
                        st.markdown(f"""
                        <div class="result-box">
                            <strong>What this means for you:</strong><br><br>
                            {explanation}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                        
                except requests.exceptions.ConnectionError:
                    st.error(
                        "⚠️ Cannot connect to the API server. "
                        "Please make sure the backend is running: "
                        "`uvicorn app.api:app --reload`"
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Disclaimer
st.markdown("""
<p class="disclaimer">
    ⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational purposes only. 
    It does not provide medical diagnoses. Always consult a qualified healthcare professional 
    for medical advice and interpretation of medical images.
</p>
""", unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **ExplainMyXray** uses AI to translate complex 
    medical imaging terms into simple language 
    that patients can understand.
    
    **How it works:**
    1. Upload your chest X-ray
    2. Our AI analyzes the image
    3. Get a simple explanation
    
    **Privacy:**
    - Images are processed locally
    - No data is stored
    - Fully offline capable
    """)
    
    # API Status
    st.header("🔌 API Status")
    if check_api_health():
        st.success("✅ Connected & Model Ready")
    else:
        st.warning("⚠️ API not connected")
        st.code("uvicorn app.api:app --reload", language="bash")
