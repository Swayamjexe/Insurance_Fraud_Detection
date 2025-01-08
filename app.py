import streamlit as st
import pandas as pd
from io import BytesIO
from src.pipeline.predict_pipeline import PredictionPipeline

# Define the paths for the model and preprocessor
MODEL_PATH = "artifacts/model.pkl"
PREPROCESSOR_PATH = "artifacts/proprocessor.pkl"

# Initialize the prediction pipeline
pipeline = PredictionPipeline(model_path=MODEL_PATH, preprocessor_path=PREPROCESSOR_PATH)

# Streamlit app
st.title("Insurance Fraud Detection - Batch Prediction")
st.write("Upload a `.csv` or `.xlsx` file with your data to detect potential insurance fraud cases.")

# File upload
uploaded_file = st.file_uploader("Upload your file (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Load the uploaded data
        if uploaded_file.name.endswith(".csv"):
            input_data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            input_data = pd.read_excel(uploaded_file)

        st.write("### Uploaded Data:")
        st.dataframe(input_data.head())


        # Predict using the pipeline
        predictions = pipeline.predict(input_data)

        # Format results
        predictions_df = pd.DataFrame({
            "Prediction": ["Fraudulent" if pred == 1 else "Legitimate" for pred in predictions]
        })

        # Combine results with input data
        output_data = pd.concat([input_data.reset_index(drop=True), predictions_df], axis=1)

        st.write("### Predictions:")
        st.dataframe(output_data)

        # Convert DataFrame to Excel for download
        def convert_df_to_excel(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Predictions")
            return output.getvalue()

        st.download_button(
            label="Download Predictions (Excel)",
            data=convert_df_to_excel(output_data),
            file_name="insurance_fraud_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.error(f"Error processing the file: {str(e)}")

