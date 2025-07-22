from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# 모델과 스케일러 로드
model = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# 홈 페이지 - UI 표시
@app.route('/')
def home():
    return render_template('index.html')

# 예측 처리
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 폼 데이터 받기
        features = [float(request.form[f]) for f in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        scaled = scaler.transform([features])
        prediction = model.predict(scaled)[0]

        species = {0: "setosa", 1: "versicolor", 2: "virginica"}
        result = species[prediction]

        return render_template('index.html', prediction_text=f"예측된 품종: {result}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"오류 발생: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
