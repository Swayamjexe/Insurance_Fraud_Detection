import streamlit as st
import pandas as pd
from io import BytesIO
import numpy as np
from src.pipeline.predict_pipeline import PredictionPipeline

# Set page config
st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .upload-text {
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize the prediction pipeline
@st.cache_resource
def load_pipeline():
    return PredictionPipeline(
        model_path="artifacts/model.pkl",
        preprocessor_path="artifacts/preprocessor.pkl"
    )

pipeline = load_pipeline()

# App title and description
st.title("🔍 Insurance Fraud Detection System")
st.markdown("""
    ### Upload your insurance claims data for fraud detection
    This system analyzes insurance claims and predicts potential fraudulent cases.
    
    **Required Format:**
    - Upload a CSV or Excel file with the same structure as the training data
    - The file should contain all necessary claim information
""")

# File upload section
st.markdown("### 📤 Upload Claims Data")
uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xlsx"],
    help="Upload a file containing insurance claims data"
)

if uploaded_file:
    try:
        # Load data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Display sample of uploaded data
        st.markdown("### 📊 Preview of Uploaded Data")
        st.dataframe(df.head())
        
        # Display data info
        st.markdown("### ℹ️ Data Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Total Records: {len(df)}")
        with col2:
            st.write(f"Total Features: {df.shape[1]}")
        
        # Make predictions
        if st.button("🔍 Analyze Claims"):
            with st.spinner("Analyzing claims..."):
                # Get predictions
                results = pipeline.predict(df)
                
                # Combine with original data
                output_data = pd.concat([df, results], axis=1)
                
                # Display results
                st.markdown("### 🎯 Analysis Results")
                
                # Summary metrics
                col1, col2 = st.columns(2)
                with col1:
                    fraudulent = (results['Prediction'] == 'Fraudulent').sum()
                    st.metric("Fraudulent Claims Detected", fraudulent)
                with col2:
                    legitimate = (results['Prediction'] == 'Legitimate').sum()
                    st.metric("Legitimate Claims", legitimate)
                
                # Display detailed results
                st.dataframe(output_data)
                
                # Download buttons
                st.markdown("### 📥 Download Results")
                
                # Excel download
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    output_data.to_excel(writer, index=False, sheet_name='Predictions')
                
                st.download_button(
                    label="📥 Download Full Results (Excel)",
                    data=excel_buffer.getvalue(),
                    file_name="fraud_detection_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"❌ Error processing the file: {str(e)}")
        st.markdown("Please ensure your file contains all required columns and correct data formats.")

# Add footer
st.markdown("---")
st.markdown("### 📋 Required Columns")
st.markdown("""
    Your input file should contain the following columns:
    - Policy information (policy_state, policy_number, etc.)
    - Insured person details (age, occupation, etc.)
    - Incident information (type, severity, location, etc.)
    - Vehicle details (make, model, year)
""")