import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from pydub import AudioSegment
import speech_recognition as sr
import io

# Load model and tokenizer from local fine-tuned directory
MODEL_PATH = "Tufan1/BioMedLM-Cardio-Fold2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Dictionaries to decode user inputs
gender_map = {1: "Female", 2: "Male"}
cholesterol_map = {1: "Normal", 2: "High", 3: "Extreme"}
glucose_map = {1: "Normal", 2: "High", 3: "Extreme"}
binary_map = {0: "No", 1: "Yes"}

# Function to predict diagnosis using the LLM
def get_prediction(age, gender, height, weight, ap_hi, ap_lo,
                   cholesterol, glucose, smoke, alco, active):
    input_text = f"""Patient Record:
- Age: {age} years
- Gender: {gender_map[gender]}
- Height: {height} cm
- Weight: {weight} kg
- Systolic BP: {ap_hi} mmHg
- Diastolic BP: {ap_lo} mmHg
- Cholesterol Level: {cholesterol_map[cholesterol]}
- Glucose Level: {glucose_map[glucose]}
- Smokes: {binary_map[smoke]}
- Alcohol Intake: {binary_map[alco]}
- Physically Active: {binary_map[active]}

Diagnosis:"""

    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=4)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    diagnosis = decoded.split("Diagnosis:")[-1].strip()
    return diagnosis

# Function to extract patient features from a phrase or transcribed audio
def extract_details_from_text(text):
    age = int(re.search(r'(\d+)\s*year', text).group(1)) if re.search(r'(\d+)\s*year', text) else None
    gender = 2 if "man" in text.lower() else (1 if "female" in text.lower() else None)
    height = int(re.search(r'(\d+)\s*cm', text).group(1)) if re.search(r'(\d+)\s*cm', text) else None
    weight = int(re.search(r'(\d+)\s*kg', text).group(1)) if re.search(r'(\d+)\s*kg', text) else None
    bp_match = re.search(r'BP\s*(\d+)[/](\d+)', text)
    ap_hi, ap_lo = (int(bp_match.group(1)), int(bp_match.group(2))) if bp_match else (None, None)
    cholesterol = 3 if "peak" in text.lower() else 2 if "elevated" in text.lower() else 1
    glucose = 3 if "extreme" in text.lower() else 2 if "high" in text.lower() else 1
    smoke = 1 if "smoke" in text.lower() else 0
    alco = 1 if "alcohol" in text.lower() else 0
    active = 1 if "exercise" in text.lower() or "active" in text.lower() else 0
    return age, gender, height, weight, ap_hi, ap_lo, cholesterol, glucose, smoke, alco, active

# Streamlit UI
st.set_page_config(page_title="Cardiovascular Disease Predictor", layout="centered")
st.title("🫀 Cardiovascular Disease Predictor (LLM Powered)")
st.markdown("This tool uses a fine-tuned BioMedLM model to predict cardiovascular conditions from structured, text, or voice input.")

input_mode = st.radio("Choose input method:", ["Manual Input", "Text Phrase", "Audio Upload"])

if input_mode == "Manual Input":
    age = st.number_input("Age (years)", min_value=1, max_value=120)
    gender = st.selectbox("Gender", [("Female", 1), ("Male", 2)], format_func=lambda x: x[0])[1]
    height = st.number_input("Height (cm)", min_value=50, max_value=250)
    weight = st.number_input("Weight (kg)", min_value=10, max_value=200)
    ap_hi = st.number_input("Systolic BP", min_value=80, max_value=250)
    ap_lo = st.number_input("Diastolic BP", min_value=40, max_value=150)
    cholesterol = st.selectbox("Cholesterol", [("Normal", 1), ("High", 2), ("Extreme", 3)], format_func=lambda x: x[0])[1]
    glucose = st.selectbox("Glucose", [("Normal", 1), ("High", 2), ("Extreme", 3)], format_func=lambda x: x[0])[1]
    smoke = st.radio("Smoker?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    alco = st.radio("Alcohol Intake?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    active = st.radio("Physically Active?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

    if st.button("Predict Diagnosis"):
        diagnosis = get_prediction(age, gender, height, weight, ap_hi, ap_lo,
                                   cholesterol, glucose, smoke, alco, active)
        st.success(f"🩺 **Predicted Diagnosis:** {diagnosis}")

elif input_mode == "Text Phrase":
    phrase = st.text_area("Enter patient details in natural language:", height=200)
    if st.button("Extract & Predict"):
        try:
            values = extract_details_from_text(phrase)
            if all(v is not None for v in values):
                diagnosis = get_prediction(*values)
                st.success(f"🩺 **Predicted Diagnosis:** {diagnosis}")
            else:
                st.warning("Couldn't extract all fields from the text. Please revise.")
        except Exception as e:
            st.error(f"Error: {e}")

elif input_mode == "Audio Upload":
    uploaded_file = st.file_uploader("Upload audio file (WAV, MP3, M4A)", type=["wav", "mp3", "m4a"])
    if uploaded_file:
        st.audio(uploaded_file, format='audio/wav')
        audio = AudioSegment.from_file(uploaded_file)
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio_data)
            st.markdown(f"**Transcribed Text:** _{text}_")
            values = extract_details_from_text(text)
            if all(v is not None for v in values):
                diagnosis = get_prediction(*values)
                st.success(f"🩺 **Predicted Diagnosis:** {diagnosis}")
            else:
                st.warning("Could not extract complete information from audio.")
        except Exception as e:
            st.error(f"Audio processing error: {e}")
