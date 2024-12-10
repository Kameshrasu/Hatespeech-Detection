import streamlit as st
import joblib
import re
import pandas as pd
from datetime import datetime
import os

# Define the paths to the files
model_path = os.path.join(os.getcwd(), 'model.pkl')
vectorizer_path = os.path.join(os.getcwd(), 'vectorizer.pkl')




# Basic page config without icon
st.set_page_config(
    page_title="Hate Speech Detector",
    layout="wide"
)

# Add custom CSS with enhanced title styling
st.markdown("""
    <style>
    .title-box {
        background: linear-gradient(to right, #1E88E5, #64B5F6);
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .title {
        color: #e6205b;
        font-size: 36px;
        font-weight: bold;
        border: 3px solid #e6205b;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
    }        
    .title-text {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        background: linear-gradient(to right, #FFFFFF, #E3F2FD);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stTextInput > div > div > input {
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Load model and vectorizer
try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    

except Exception as e:
    st.error(f"Error loading model files: {str(e)}")
    st.stop()

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

def analyze_text(text):
    processed_text = preprocess_text(text)
    vector = vectorizer.transform([processed_text])
    prediction = model.predict(vector)
    proba = model.predict_proba(vector)[0]
    confidence = proba.max() * 100
    print(prediction)
    return prediction[0], confidence

# Title with gradient box
st.markdown("""
    
        <h1 class="title">
            Hate Speech Detection
        </h1>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["Text Analysis", "File Upload", "History"])

with tab1:
    # Text input area with padding
    st.markdown("<br>", unsafe_allow_html=True)
    user_input = st.text_area(
        "Enter text to analyze:",
        height=100,
        placeholder="Type or paste your text here..."
    )
    
    # Analyze button
    if st.button("Analyze Text", type="primary", key="analyze_text"):
        if user_input:
            try:
                # Show processing message
                with st.spinner('Analyzing...'):
                    # Get prediction and confidence
                    prediction, confidence = analyze_text(user_input)
                    
                    # Store in history
                    st.session_state.history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'text': user_input[:100] + "..." if len(user_input) > 100 else user_input,
                        'prediction': "Hate Speech" if prediction == 1 else "Clean",
                        'confidence': f"{confidence:.2f}%"
                    })
                    
                    # Display results
                    if prediction == 1:
                        st.error("⚠️ Hate Speech Detected!")
                        st.write(f"Confidence: {confidence:.2f}%")
                    else:
                        st.success("✅ No Hate Speech Detected")
                        st.write(f"Confidence: {confidence:.2f}%")
                            
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
        else:
            st.warning("Please enter some text to analyze.")

with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Upload Text File")
    uploaded_file = st.file_uploader("Choose a text file", type=['txt'])
    
    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode()
            st.text_area("File Content:", value=content, height=150)
            if st.button("Analyze File", type="primary"):
                prediction, confidence = analyze_text(content)
                if prediction == 1:
                    st.error("⚠️ Hate Speech Detected in File!")
                    st.write(f"Confidence: {confidence:.2f}%")
                else:
                    st.success("✅ No Hate Speech Detected in File")
                    st.write(f"Confidence: {confidence:.2f}%")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Analysis History")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        
        # Export history
        if st.button("Export History to CSV"):
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="hate_speech_analysis_history.csv",
                mime="text/csv"
            )
    else:
        st.info("No analysis history yet. Start analyzing text to build history.")

# Footer
st.markdown("---")
st.markdown("""
    <p style='text-align: center; color: #666;'>
        Hate Speech Detection System | Created with Streamlit
    </p>
""", unsafe_allow_html=True)
