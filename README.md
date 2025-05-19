# 🫀 LLM-driven Smart Screening Tool for Cardiovascular Health Monitoring through Multimodal Interface

## 📌 Project Objective

To develop a rapid, scalable, accessible, and trustworthy AI-based smart screening tool for personalized cardiovascular health monitoring and early risk detection.

## 🚀 Features

- 🔬 Fine-tuned `Bio_ClinicalBERT` model for binary classification (`cardio`: 0 = No CVD, 1 = CVD)
- 🎙️ Multi-modal input support: Numeric, text, and voice
- 📊 Displays extracted features, missing fields, and prediction confidence
- 📈 LoRA + K-Fold cross-validation fine-tuning for robustness
- 🌐 Deployed via Hugging Face Spaces + remote huggingface CPU

---

## 🧠 Technologies Used

- `Transformers` (`Bio_ClinicalBERT`)
- `LoRA` for efficient fine-tuning
- `scikit-learn`, `pandas`, `matplotlib`
- `Streamlit` for the frontend
- `Hugging Face` for model hosting and inference
- `SpeechRecognition` + `pydub` for audio input

---

## 🗃️ Project Structure

![image](https://github.com/user-attachments/assets/70b97d66-04a7-4002-8a80-23ae0faa87b9)


---

## Model Hosting

LoRA-adapted version - "Tufan1/BioClinicalBERT-Cardio-Classifier-Fold-dummy-Final10"

Training: 70,000 samples, 10-fold CV, uploaded

## Output
Input Prompt: "Ms. Patel, a 60 year old woman with height 176 centimeters tall and weight 54 kilograms, presented for her annual cardiac evaluation. Her blood pressure 140/90 mmHg raised concerns, particularly when combined with notably lipid profiles showing total cholesterol is peak concentrations. Recent lab work revealed fasting blood glucose is high range at 128 mg/dL. The patient acknowledged being a regular tobacco user (10 cigarettes/day) and consuming 3-4 alcohol drinks weekly. Despite her sedentary office job, she maintains moderate physical activity through daily 30-minute brisk walks. Family history reveals paternal hypertension and maternal diabetes. Current symptoms include occasional dizziness when standing, though she denies chest pain or palpitations. Dietary assessment showed high saturated fat intake, while stress levels remain manageable through yoga practice"

Output: Prediction: CVD (Confidence: 79.79%)
![image](https://github.com/user-attachments/assets/acdc290d-0aee-4a4e-90f9-fe6f7e541e70)


## 🧪 Evaluation Metrics 

![image](https://github.com/user-attachments/assets/a612dc4e-db4a-48c3-888a-f7f7f5e0c01a)

![image](https://github.com/user-attachments/assets/ed3f2807-3863-4516-983d-4031d02b35f4)



## 📜 License
This project is open-source for academic use. Contact for commercial applications.

## 🙋‍♂️ Author
- Tufan Paul
- M.Tech Clinical Engineering, IIT Madras
- Email: tufanpaul2016@gmail.com
- GitHub: Tufan-2000

## 🔗 Links
- 🔬 Model on Hugging Face - Tufan1/BioClinicalBERT-Cardio-Classifier-Fold-dummy-Final10
- 🌐 Live App on Hugging Face Spaces - https://huggingface.co/spaces/Tufan1/CVD-PREDICTOR-FINAL-ClinicalBIOBERT
