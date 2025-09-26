import os
import tempfile
import streamlit as st
import pandas as pd
import PyPDF2
from gtts import gTTS
import google.generativeai as genai

# Configure Google Generative AI API
genai.configure(api_key="AIzaSyDcas9UKOBjES0EU2dRpA-F6ZaAsRNgOKE")

# Load the Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")


def generate_response(prompt: str) -> str:
    """Generate a text response from the Gemini model."""
    response = model.generate_content(prompt)
    return response.text if response and response.text else ""


def translate_text(text: str, target_language: str = "English") -> str:
    """Translate text into the target language using Gemini."""
    prompt = f"Translate the following text into {target_language}:\n\n{text}"
    response = model.generate_content(prompt)
    return response.text if response and response.text else ""


def text_to_speech(text: str, lang: str = "en") -> str:
    """
    Convert text to speech using gTTS.
    Returns the path to the temporary audio file.
    """
    tts = gTTS(text=text, lang=lang)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name


# ---- Streamlit UI ----
st.title("AI Text Translator & Audio Generator")

# Input type: text or PDF
input_option = st.radio("Input type:", ["Text", "PDF File"])

user_text = ""
if input_option == "Text":
    user_text = st.text_area("Enter your text here:")
else:
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        user_text = ""
        for page in pdf_reader.pages:
            user_text += page.extract_text() + "\n"

# Language selection
target_language = st.selectbox(
    "Select target language for translation:",
    ["English", "French", "Spanish", "German", "Chinese"]
)

# Generate translation & response
if st.button("Generate Translation & Audio") and user_text.strip():
    with st.spinner("Processing..."):
        translated_text = translate_text(user_text, target_language)
        audio_file_path = text_to_speech(translated_text)
    
    st.subheader("Translated Text:")
    st.write(translated_text)
    
    st.audio(audio_file_path, format="audio/mp3")
    
    # Provide download link
    st.download_button(
        label="Download Audio",
        data=open(audio_file_path, "rb").read(),
        file_name="translation.mp3",
        mime="audio/mp3"
    )
