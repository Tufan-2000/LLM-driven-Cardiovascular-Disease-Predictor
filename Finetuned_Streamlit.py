import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load fine-tuned model
@st.cache_resource
def load_model():
    model_path = "./BioMedLM-Cardio"  # Path to saved model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model

tokenizer, model = load_model()

# Dictionaries for mapping
gender_map = {1: "Female", 2: "Male"}
cholesterol_map = {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}
glucose_map = {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}
binary_map = {0: "No", 1: "Yes"}

# Prediction function
def get_prediction(inputs):
    input_text = f"""Patient Record:
- Age: {inputs['age']} years
- Gender: {gender_map[inputs['gender']]}
- Height: {inputs['height']} cm
- Weight: {inputs['weight']} kg
- Systolic BP: {inputs['ap_hi']} mmHg
- Diastolic BP: {inputs['ap_lo']} mmHg
- Cholesterol Level: {cholesterol_map[inputs['cholesterol']]}
- Glucose Level: {glucose_map[inputs['glucose']]}
- Smokes: {binary_map[inputs['smoke']]}
- Alcohol Intake: {binary_map[inputs['alco']]}
- Physically Active: {binary_map[inputs['active']]}

Diagnosis:"""

    encoded = tokenizer(input_text, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        output = model.generate(**encoded, max_new_tokens=4)
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result.split("Diagnosis:")[-1].strip()

# Streamlit UI
st.title("🫀 Cardiovascular Disease Predictor (BioMedLM)")

with st.form("patient_form"):
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=45)
    gender = st.selectbox("Gender", [1, 2], format_func=lambda x: gender_map[x])
    height = st.number_input("Height (cm)", value=165)
    weight = st.number_input("Weight (kg)", value=70)
    ap_hi = st.number_input("Systolic BP", value=120)
    ap_lo = st.number_input("Diastolic BP", value=80)
    cholesterol = st.selectbox("Cholesterol Level", [1, 2, 3], format_func=lambda x: cholesterol_map[x])
    glucose = st.selectbox("Glucose Level", [1, 2, 3], format_func=lambda x: glucose_map[x])
    smoke = st.selectbox("Smoker?", [0, 1], format_func=lambda x: binary_map[x])
    alco = st.selectbox("Alcohol Intake?", [0, 1], format_func=lambda x: binary_map[x])
    active = st.selectbox("Physically Active?", [0, 1], format_func=lambda x: binary_map[x])

    submitted = st.form_submit_button("Predict")

    if submitted:
        inputs = {
            "age": age, "gender": gender, "height": height, "weight": weight,
            "ap_hi": ap_hi, "ap_lo": ap_lo, "cholesterol": cholesterol,
            "glucose": glucose, "smoke": smoke, "alco": alco, "active": active
        }
        diagnosis = get_prediction(inputs)
        st.success(f"🩺 Predicted Diagnosis: **{diagnosis}**")
