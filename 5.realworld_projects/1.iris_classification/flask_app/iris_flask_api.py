# iris_flask_api.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Flask 앱 생성
app = Flask(__name__)

# 모델 및 전처리기 불러오기
model = joblib.load("models/rf_model.pkl")
scaler = StandardScaler()

# Iris 특성명 (훈련 때 사용한 순서와 동일하게)
feature_names = ['sepal length (cm)', 'sepal width (cm)', 
                 'petal length (cm)', 'petal width (cm)']

# 예측 API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # JSON에서 입력 받기
        input_data = request.get_json()
        features = input_data.get("features")

        # 입력 유효성 검사
        if not features or len(features) != 4:
            return jsonify({"error": "입력값은 4개의 특성값이 있어야 합니다."}), 400

        # DataFrame으로 변환
        input_df = pd.DataFrame([features], columns=feature_names)

        # 스케일링 적용 (동일한 방식으로 재학습 필요 시 joblib.dump(scaler, "scaler.pkl"))
        input_scaled = scaler.fit_transform(input_df)  # fit_transform -> 실제 배포시엔 transform만 사용

        # 예측
        prediction = model.predict(input_scaled)

        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 서버 실행
if __name__ == "__main__":
    app.run(debug=True)
