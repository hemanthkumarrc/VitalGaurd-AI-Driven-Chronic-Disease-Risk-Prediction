from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load training data
df = pd.read_csv("Training.csv")
disease_mapping = {disease: idx for idx, disease in enumerate(df["prognosis"].unique())}
df.replace({'prognosis': disease_mapping}, inplace=True)

# Features and labels
l1 = df.columns[:-1].tolist()  # Symptoms list
disease = list(disease_mapping.keys())

X = df[l1]
y = np.ravel(df[["prognosis"]])

# Train model
gnb = MultinomialNB()
gnb.fit(X, y)


# Below is testing part, already tested once only test again if data is changed.

# # Load test data for evaluation
# tr = pd.read_csv("Testing.csv")
# tr.replace({'prognosis': disease_mapping}, inplace=True)

# X_test = tr[l1]
# y_test = np.ravel(tr[["prognosis"]])

# # Model Evaluation
# y_pred = gnb.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = data.get("symptoms", [])

    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    symptom_vector = [1 if symptom in symptoms else 0 for symptom in l1]
    prediction = gnb.predict([symptom_vector])[0]
    predicted_disease = disease[prediction]

    return jsonify({"prediction": predicted_disease})

if __name__ == '__main__':
    app.run(debug=True)
