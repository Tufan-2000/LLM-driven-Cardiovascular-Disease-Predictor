import os #
import io
import re
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pydub import AudioSegment
import speech_recognition as sr
from audiorecorder import audiorecorder
import shap
import numpy as np

# Streamlit config
st.set_page_config(page_title="Cardio Disease Predictor", layout="wide")
# ⬇️ Insert the reset logic right after page config
if st.button("🔄 Reset All"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state["input_mode"] = "Manual Input"  # Reset radio button to default
    st.rerun()

st.title("🫀 LLM Powered Cardiovascular Disease Predictor")

# Force CPU
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device("cpu")

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_name = "Tufan1/BioClinicalBERT-Cardio-Classifier-Fold-per1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(DEVICE)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Mappings
MEDICAL_MAPPINGS = {
    'gender': {1: "Female", 2: "Male"},
    'cholesterol': {1: "Normal", 2: "Elevated", 3: "Peak"},
    'glucose': {1: "Normal", 2: "High", 3: "Extreme"},
    'binary': {0: "No", 1: "Yes"}
}

REVERSE_MAPPINGS = {
    'gender': {"Female": 1, "Male": 2},
    'cholesterol': {"Normal": 1, "Elevated": 2, "Peak": 3},
    'glucose': {"Normal": 1, "High": 2, "Extreme": 3},
    'binary': {"No": 0, "Yes": 1}
}

# Feature order for SHAP
FEATURE_ORDER = [
    'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
    'cholesterol', 'glucose', 'smoke', 'alco', 'active'
]

# SHAP Model Wrapper
class ShapModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, X):
        preds = []
        for row in X:
            try:
                # Ensure all values are integers or valid for mapping
                feature_dict = {}
                for k, v in zip(FEATURE_ORDER, row):
                    # Convert to int for safety, handle potential float values from SHAP perturbations
                    if k in ['age', 'height', 'weight', 'ap_hi', 'ap_lo']:
                        feature_dict[k] = int(round(float(v))) if v is not None else 0
                    else:
                        # For categorical features, ensure value is within valid range for mapping
                        if k == 'gender':
                            feature_dict[k] = max(1, min(2, int(round(float(v))))) if v is not None else 1
                        elif k in ['cholesterol', 'glucose']:
                            feature_dict[k] = max(1, min(3, int(round(float(v))))) if v is not None else 1
                        else:  # binary fields
                            feature_dict[k] = max(0, min(1, int(round(float(v))))) if v is not None else 0
                
                # Create input text with validated values
                input_text = f"""Patient Record:
- Age: {feature_dict['age']} years
- Gender: {MEDICAL_MAPPINGS['gender'][feature_dict['gender']]}
- Height: {feature_dict['height']} cm
- Weight: {feature_dict['weight']} kg
- BP: {feature_dict['ap_hi']}/{feature_dict['ap_lo']} mmHg
- Cholesterol: {MEDICAL_MAPPINGS['cholesterol'][feature_dict['cholesterol']]}
- Glucose: {MEDICAL_MAPPINGS['glucose'][feature_dict['glucose']]}
- Smoke: {MEDICAL_MAPPINGS['binary'][feature_dict['smoke']]}
- Alco: {MEDICAL_MAPPINGS['binary'][feature_dict['alco']]}
- Active: {MEDICAL_MAPPINGS['binary'][feature_dict['active']]}"""
                
                inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
                with torch.no_grad():
                    prob = torch.sigmoid(self.model(**inputs).logits).item()
                preds.append(prob)
            except Exception as e:
                # Log error for debugging but don't stop the process
                # st.error(f"Error in SHAP prediction: {str(e)}")  # Comment out to avoid cluttering UI
                preds.append(0.5)  # Default to neutral probability on error
        return np.array(preds)


# Initialize SHAP Explainer
@st.cache_resource
def get_shap_explainer(_model, _tokenizer):
    wrapper = ShapModelWrapper(_model, _tokenizer)
    masker = shap.maskers.Independent(np.zeros((1, len(FEATURE_ORDER))))
    return shap.Explainer(wrapper, masker, algorithm="permutation", max_evals=50)

shap_explainer = get_shap_explainer(model, tokenizer)

# Feature extraction from text
def safe_extract(pattern, text, group=1, default=None):
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(group) if match else default

def text_to_features(text):
    patterns = {
        'age': r'(\d+)\s*(?:years?|yrs?|year-old|year old|y)',
        'gender': r'\b(male|female|man|woman)\b',
        'height': r'(?:height|ht|tall)\D*(\d+)\s*(?:cm|centimeters?)',
        'weight': r'(?:weight|wt|weighs|weighing)\D*(\d+)\s*(?:kg|kilos?|kilograms?)',
        'bp': r'(?:blood pressure|bp)\D*(\d+)\s*/\s*(\d+)\s*(?:mmHg|)',
        'cholesterol': r'(cholest(?:erol)?|lipids?)\s*(?:is|level)?\s*(normal|elevated|peak)',
        'glucose': r'(glucose|sugar|blood sugar)\s*(?:is|level)?\s*(normal|high|extreme)',
        'smoke': r'\b(smokes?|smoking|tobacco|(?:non-?smoker)|no tobacco|never smoked|cigarettes?|smoke)\b',
        'alco': r'\b(alcohol|drinks?|(?:no alcohol)|never drinks?|teetotal|abstains|beer|wine|spirits)\b',
        'active': r'\b(active|exercises?|sedentary|physically fit|no exercise|inactive|gym|walking|jogging)\b'
    }
    def interpret_lifestyle(match, param):
        """Handle negations and synonyms for lifestyle factors"""
        if not match:
            return None
        match = match.lower()
        negation_terms = {
            'smoke': ['non', 'no', 'never'],
            'alco': ['non', 'no', 'never', 'abstain', 'teetotal'],
            'active': ['sedentary', 'no exercise', 'inactive']
        }
        if any(n in match for n in negation_terms.get(param, [])):
            return 0
        return 1

    features = {
        'age': int(safe_extract(patterns['age'], text, 1)) if safe_extract(patterns['age'], text, 1) else None,
        'gender': 2 if safe_extract(patterns['gender'], text, 1, '').lower() in ['male', 'man'] else (
            1 if safe_extract(patterns['gender'], text, 1, '').lower() in ['female', 'woman'] else None),
        'height': int(safe_extract(patterns['height'], text, 1)) if safe_extract(patterns['height'], text, 1) else None,
        'weight': int(safe_extract(patterns['weight'], text, 1)) if safe_extract(patterns['weight'], text, 1) else None,
        'ap_hi': None,
        'ap_lo': None,
        'cholesterol': REVERSE_MAPPINGS['cholesterol'].get(safe_extract(patterns['cholesterol'], text, 2, '').capitalize()) if safe_extract(patterns['cholesterol'], text, 2) else None,
        'glucose': REVERSE_MAPPINGS['glucose'].get(safe_extract(patterns['glucose'], text, 2, '').capitalize()) if safe_extract(patterns['glucose'], text, 2) else None,
        'smoke': 1 if safe_extract(patterns['smoke'], text) else (0 if 'smoke' in text.lower() else None),
        'alco': 1 if safe_extract(patterns['alco'], text) else (0 if 'alcohol' in text.lower() else None),
        'active': 1 if safe_extract(patterns['active'], text) else (0 if 'active' in text.lower() else None),
    }

    # BP
    bp_match = re.search(patterns['bp'], text, re.IGNORECASE)
    if bp_match:
        features['ap_hi'], features['ap_lo'] = map(int, bp_match.groups())

    # Cholesterol/glucose
    chol = safe_extract(patterns['cholesterol'], text, 2)
    if chol: features['cholesterol'] = REVERSE_MAPPINGS['cholesterol'].get(chol.capitalize(), 1)

    gluc = safe_extract(patterns['glucose'], text, 2)
    if gluc: features['glucose'] = REVERSE_MAPPINGS['glucose'].get(gluc.capitalize(), 1)

    # Lifestyle factors with negation handling
    for param in ['smoke', 'alco', 'active']:
        match = safe_extract(patterns[param], text, 0)  # Get full match
        features[param] = interpret_lifestyle(match, param)
        
    return features

# Audio processing
def transcribe_audio(audio_bytes):
    recognizer = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
        audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)

# Prediction
def predict(features):
    input_text = f"""Patient Record:
- Age: {features['age']} years
- Gender: {MEDICAL_MAPPINGS['gender'][features['gender']]}
- Height: {features['height']} cm
- Weight: {features['weight']} kg
- BP: {features['ap_hi']}/{features['ap_lo']} mmHg
- Cholesterol: {MEDICAL_MAPPINGS['cholesterol'][features['cholesterol']]}
- Glucose: {MEDICAL_MAPPINGS['glucose'][features['glucose']]}
- Smoke: {MEDICAL_MAPPINGS['binary'][features['smoke']]}
- Alco: {MEDICAL_MAPPINGS['binary'][features['alco']]}
- Active: {MEDICAL_MAPPINGS['binary'][features['active']]}"""

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(model(**inputs).logits).item()
    return ("Cardiovascular Disease" if prob >= 0.5 else "No Cardiovascular Disease", round(prob * 100, 2))

# Manual input form for filling missing fields
def fill_missing(features):
    st.markdown("### 🧩 Please complete the missing information below")
    filled_values = {}
    with st.form("fill_missing_form"):
        for field in features:
            if features[field] is None:
                if field in ['age', 'height', 'weight', 'ap_hi', 'ap_lo']:
                    filled_values[field] = st.number_input(f"{field.replace('_', ' ').title()}:", 
                                                          min_value=1, step=1, value=1)
                elif field == 'gender':
                    filled_values[field] = REVERSE_MAPPINGS['gender'][
                        st.selectbox("Gender", ["Female", "Male"])
                    ]
                elif field == 'cholesterol':
                    filled_values[field] = REVERSE_MAPPINGS['cholesterol'][
                        st.selectbox("Cholesterol", ["Normal", "Elevated", "Peak"])
                    ]
                elif field == 'glucose':
                    filled_values[field] = REVERSE_MAPPINGS['glucose'][
                        st.selectbox("Glucose", ["Normal", "High", "Extreme"])
                    ]
                else:  # binary fields
                    filled_values[field] = REVERSE_MAPPINGS['binary'][
                        st.selectbox(field.capitalize(), ["Yes", "No"])
                    ]
        submitted = st.form_submit_button("Submit & Predict")
    
    if submitted:
        # Update original features with user-filled values
        features.update(filled_values)
        return True, features
    return False, features

    
# Display features
def show_features(features):
    st.markdown("### 🧾 Extracted Features:")
    st.json({k: (v if v is not None else "❌ Missing") for k, v in features.items()})

# ================== MAIN ==================
if "features" not in st.session_state:
    st.session_state.features = {}
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

input_type = st.radio(
    "Select input method:",
    ["Manual Input", "Text Input", "Audio Input"],
    key="input_mode"
)

if input_type == "Manual Input":
    with st.form("manual_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (years)", 10, 120)
            height = st.number_input("Height (cm)", 50, 250)
            ap_hi = st.number_input("Systolic BP", 80, 250)
            cholesterol = st.selectbox("Cholesterol", ["Normal", "Elevated", "Peak"])
            smoke = st.selectbox("Smoke", ["Yes", "No"])
            active = st.selectbox("Active", ["Yes", "No"])
        with col2:
            gender = st.selectbox("Gender", ["Female", "Male"])
            weight = st.number_input("Weight (kg)", 30, 300)
            ap_lo = st.number_input("Diastolic BP", 40, 150)
            glucose = st.selectbox("Glucose", ["Normal", "High", "Extreme"])
            alco = st.selectbox("Alco", ["Yes", "No"])
        submitted = st.form_submit_button("Predict")

        if submitted:
            st.session_state.features = {
                'age': age, 'gender': REVERSE_MAPPINGS['gender'][gender],
                'height': height, 'weight': weight,
                'ap_hi': ap_hi, 'ap_lo': ap_lo,
                'cholesterol': REVERSE_MAPPINGS['cholesterol'][cholesterol],
                'glucose': REVERSE_MAPPINGS['glucose'][glucose],
                'smoke': REVERSE_MAPPINGS['binary'][smoke],
                'alco': REVERSE_MAPPINGS['binary'][alco],
                'active': REVERSE_MAPPINGS['binary'][active]
            }
            st.session_state.prediction_result = predict(st.session_state.features)
            st.session_state.prediction_done = True

if input_type == "Text Input":
    example_text = (
        "A 60 year old female having height 176 cm & weight 54 kg, "
        "with BP 140/90, glucose is high. She used to take alcohol, "
        "physically active, also having cholesterol normal."
    )
    text_input = st.text_area("Enter patient description:", height=150, placeholder=example_text)
    if st.button("Extract Features"):
        st.session_state.features = text_to_features(text_input)
        st.session_state.prediction_done = False

if input_type == "Audio Input":
    audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "ogg"])
    recorded_audio = audiorecorder("Start Recording", "Stop Recording")

    if st.button("Transcribe & Extract"):
        try:
            if recorded_audio and len(recorded_audio) > 0:
                audio_io = io.BytesIO()
                recorded_audio.export(audio_io, format="wav")
                audio_bytes = audio_io.getvalue()
            elif audio_file is not None:
                audio_bytes = audio_file.read()
            else:
                st.warning("No audio provided.")
                st.stop()

            transcribed = transcribe_audio(audio_bytes)
            st.markdown("**Transcribed Text:**")
            st.info(transcribed)
            st.session_state.features = text_to_features(transcribed)
            st.session_state.prediction_done = False
        except Exception as e:
            st.error(f"Error: {e}")

# Proceed to show extracted features and prediction logic
features = st.session_state.features
if features:
    show_features(features)

    # Check for missing values
    missing_keys = [k for k, v in features.items() if v is None]

    if missing_keys:
        submit_missing_inputs, filled_values = fill_missing(features)
        if submit_missing_inputs:
            # Update features with the newly filled values
            features.update(filled_values)
            st.session_state.features = features
            st.session_state.prediction_result = predict(features)
            st.session_state.prediction_done = True   
            st.rerun()  # Ensures the UI updates and shows the new features

    # NEW: Add Predict button when all features are present
    elif not st.session_state.prediction_done:
        if st.button("Predict"):  # <-- This is the critical addition
            st.session_state.prediction_result = predict(features)
            st.session_state.prediction_done = True

# Show final result if available
if st.session_state.prediction_result:
    prediction, confidence = st.session_state.prediction_result
    st.success(f"**Prediction:** {prediction} (Probability: {confidence}%)")
    # --- SHAP Explanation Block ---
    with st.spinner("🔍 Analyzing contributing features (20-30 seconds)..."):
        try:
            shap_input = np.array([[features[k] for k in FEATURE_ORDER]], dtype=float)
            shap_values = shap_explainer(shap_input)
            impacts = {FEATURE_ORDER[i]: float(shap_values.values[0][i]) for i in range(len(FEATURE_ORDER))}
            top_features = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        # Estimate base value (expected value)
            base_value = float(shap_explainer.expected_value) if hasattr(shap_explainer, 'expected_value') else 0.5
            base_prob = round(base_value * 100, 2)
        
            st.markdown("### 🔍 Top 3 Contributing Features")
            st.markdown(f"**Base Probability (without specific features):** {base_prob}%")
            total_impact = 0
            for feat, val in top_features:
                direction = "↑ Increases risk" if val > 0 else "↓ Decreases risk"
                impact_pct = abs(val) * 100
                st.markdown(f"- **{feat.replace('_', ' ').title()}**: {direction} (Impact: {impact_pct:.2f}%)")
            st.markdown("*Note: Model predictions are based on training data patterns and may not always align with clinical expectations due to feature interactions or data specifics.*")
        except Exception as e:
            st.error(f"Explanation failed: {str(e)}")


    
    # 👉 ADD HEALTH SUGGESTION BASED ON PROBABILITY
    def get_health_suggestion(probability):
        if probability < 50:
            return ("🟢 *You are healthy.*\n\n"
                "- Keep maintaining a healthy lifestyle 🌟\n"
                "- Stay active and eat a balanced diet 🍎\n"
                "- Routine checkups are still recommended ✅")
        elif 50 <= probability < 65:
            return ("🟡 *Mild Risk Detected.*\n\n"
                    "- Engage in regular exercise 🏃‍♂️\n"
                    "- Maintain a balanced, heart-healthy diet 🥗\n"
                    "- Monitor your health regularly 📈")
        elif 65 <= probability < 80:
            return ("🟠 *Moderate Risk Detected.*\n\n"
                    "- Be cautious about lifestyle choices ⚡\n"
                    "- Schedule regular health checkups 🩺\n"
                    "- Consult a healthcare provider for preventive advice 💬")
        elif probability >= 80:
            return ("🔴 *High Risk Detected!* ⚠️\n\n"
                    "- Consult a doctor immediately 👨‍⚕️👩‍⚕️\n"
                    "- Undergo detailed cardiovascular examinations 🏥\n"
                    "- Follow clinical advice without delay.")
        else:
            return None  # No suggestion if probability < 50%

    suggestion = get_health_suggestion(confidence)
    if suggestion:
        st.markdown("---")
        st.subheader("🔔 Health Suggestion Based on Prediction:")
        st.markdown(suggestion)
