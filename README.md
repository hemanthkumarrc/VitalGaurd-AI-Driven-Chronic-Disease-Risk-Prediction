🩺 Chronic Disease Prediction
A Flask web app that predicts chronic diseases based on selected symptoms using a Naive Bayes model.


🚀 Features
Symptom-based disease prediction

Trained on medical dataset (Training.csv)

Simple, responsive UI

⚙️ Setup
bash
Copy
Edit
git clone https://github.com/your-username/chronic-disease-prediction.git
cd chronic-disease-prediction
pip install flask pandas numpy scikit-learn
python app.py
Runs at http://127.0.0.1:5000

📊 API
POST /predict

json
{"symptoms": ["itching", "skin_rash"]}
Response

json
{"prediction": "Fungal infection"}

📌 Note
For educational purposes only — not a medical tool.
