from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# 모델과 스케일러 로드
# model = joblib.load("models/lr_model.pkl")
model = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# 홈 페이지 - UI 표시
@app.route('/')
def home():
    return render_template('index2.html')

# 예측 처리
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 폼 데이터 받기
        features = [float(request.form[f]) for f in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

        # 주의: 학습할 때 스케일링을 했다면, 추론할 때도 반드시 같은 방식으로 스케일링을 해줘야 하고,
        # 학습할 때 스케일링을 하지 않았다면, 추론할 때도 그대로 사용해야 합니다.
        
        # scaled = scaler.transform([features])  # ← ❌ 제거
        # prediction = model.predict(scaled)[0]
        
        prediction = model.predict([features])[0]  # ✔ 바로 예측

        species = {0: "setosa", 1: "versicolor", 2: "virginica"}
        result = species[prediction]

        return render_template('index.html', prediction_text=f"예측된 품종: {result}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"오류 발생: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
