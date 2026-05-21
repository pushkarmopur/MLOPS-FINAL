from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = 'model.pkl'

model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()

        for key in data:
            data[key] = float(data[key])

        df = pd.DataFrame([data])

        if hasattr(model, 'feature_names_in_'):
            df = df[model.feature_names_in_]

        prediction = model.predict(df)

        result_text = (
            "Good Quality Wine"
            if prediction[0] == 1
            else "Poor Quality Wine"
        )

        return render_template(
            "result.html",
            prediction=result_text
        )

    except Exception as e:
        return render_template(
            "result.html",
            prediction=f"Error: {str(e)}"
        )


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
