from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# 모델 로딩
# model = joblib.load("models/lr_model.pkl")
model = joblib.load("models/rf_model.pkl")

@app.route('/')
def home():
    return "<H1>Hello, Flask</H1>"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = [data[f] for f in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        print("입력값 검증: ", features)
        
        prediction = model.predict([features])[0]
        print("예측결과: ", prediction)
        
        species = {0: "Setosa", 1: "Versicolor", 2: "Verginica"}
        result = species[prediction]
        
        # return jsonify({"prediction": int(prediction)})
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)


# | 품종       | sepal_length  | sepal_width  | petal_length  | petal_width  |
# | ---------- | ------------- | ------------ | ------------- | ------------ |
# | setosa     | 4.3–5.8       | 2.3–4.4      | 1.0–1.9       | 0.1–0.6      |
# | versicolor | 4.9–7.0       | 2.0–3.4      | 3.0–5.1       | 1.0–1.8      |
# | virginica  | 4.9–7.9       | 2.2–3.8      | 4.5–6.9       | 1.4–2.5      |


# curl -X POST 127.0.0.1:5000/predict 
# -H "Content-Type: application/json" 
# -d '{"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5}'

# 1. setosa 예측
# curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"sepal_length\":5.0,\"sepal_width\":3.6,\"petal_length\":1.4,\"petal_width\":0.2}"

# 2. versicolor 예측
# curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"sepal_length\":6.0,\"sepal_width\":2.7,\"petal_length\":4.5,\"petal_width\":1.5}"

# 3. virginica 예측
# curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"sepal_length\":6.5,\"sepal_width\":3.0,\"petal_length\":5.8,\"petal_width\":2.2}"
