# 5_predict_live.py

from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# 모델 로드 (RandomForestClassifier 포함된 Pipeline)
model = joblib.load("models/rf_multiclass.pkl")

# 예측에 사용할 피처 리스트 (데이터 순서 보장용)
FEATURES = ['duration', 'packet_size', 'src_bytes', 'dst_bytes']  # 수정 필요시 교체
CLASS_LABELS = ['Normal', 'DoS', 'Probe']

@app.route("/")
def home():
    return "Network Multiclass Anomaly Predictor API"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # JSON 입력 받기
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON input received."}), 400

        # 입력값을 DataFrame으로 변환
        input_df = pd.DataFrame([data])[FEATURES]

        # 예측 수행
        pred_proba = model.predict_proba(input_df)[0]
        pred_class_index = pred_proba.argmax()
        pred_class_label = CLASS_LABELS[pred_class_index]

        # 결과 반환
        return jsonify({
            "prediction": int(pred_class_index),
            "label": pred_class_label,
            "probabilities": {CLASS_LABELS[i]: float(prob) for i, prob in enumerate(pred_proba)}
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
