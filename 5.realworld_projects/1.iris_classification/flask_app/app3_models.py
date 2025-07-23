from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# 모델 경로
MODEL_DIR = "models"
MODEL_FILES = {
    "Random Forest": "rf_model.pkl",
    "SVM": "svm_model.pkl",
    "KNN": "knn_model.pkl",
    "Naive Bayes": "nb_model.pkl"
}

# 스케일러 로드
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

# 특성 이름
FEATURE_NAMES = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    selected_model = None

    if request.method == "POST":
        try:
            # 사용자 입력 받기
            values = [float(request.form[feat]) for feat in FEATURE_NAMES]
            model_label = request.form["model"]
            selected_model = model_label

            # 모델 로드
            model_path = os.path.join(MODEL_DIR, MODEL_FILES[model_label])
            model = joblib.load(model_path)

            # 입력값 전처리
            df = pd.DataFrame([values], columns=FEATURE_NAMES)
            scaled = scaler.transform(df)

            # 예측
            pred = model.predict(scaled)[0]
            prediction = ["setosa", "versicolor", "virginica"][pred]

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index3.html",
                           feature_names=FEATURE_NAMES,
                           model_names=list(MODEL_FILES.keys()),
                           prediction=prediction,
                           selected_model=selected_model)

if __name__ == "__main__":
    app.run(debug=True)
