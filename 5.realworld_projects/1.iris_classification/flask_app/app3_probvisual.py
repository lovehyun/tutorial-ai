# flask_app/app.py
from flask import Flask, request, render_template
import joblib, os, io, base64
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 이 줄을 반드시 먼저 선언해야 함! Flask 에서 matplotlib의 GUI백엔드를 사용하지 않도록.
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

app = Flask(__name__)

# 데이터 및 모델 정의
from sklearn.datasets import load_iris
iris = load_iris()
target_names = iris.target_names.tolist()

# FEATURE_NAMES는 학습에 사용된 정확한 컬럼 이름 사용 → scaler.transform() 오류 방지
FEATURE_NAMES = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
DISPLAY_NAMES = ['꽃받침 길이', '꽃받침 너비', '꽃잎 길이', '꽃잎 너비']

model_dir = os.path.join(os.path.dirname(__file__), 'models')
model_files = {
    'RandomForest': 'rf_model.pkl',
    'SVM': 'svm_model.pkl',
    'KNN': 'knn_model.pkl',
    'NaiveBayes': 'nb_model.pkl'
}
scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))

# 확률 그래프 생성
def plot_prediction_proba(proba, target_names):
    fig, ax = plt.subplots()
    ax.bar(target_names, proba, color='skyblue')
    ax.set_title("예측 확률")
    ax.set_ylim(0, 1)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    result, img_base64 = None, None
    selected_model = 'RandomForest'

    if request.method == 'POST':
        try:
            data = [float(request.form.get(name)) for name in FEATURE_NAMES]
        except TypeError:
            return "입력값이 모두 채워졌는지 확인해주세요.", 400

        selected_model = request.form.get("model")
        df = pd.DataFrame([data], columns=FEATURE_NAMES)
        df_scaled = pd.DataFrame(scaler.transform(df), columns=FEATURE_NAMES)

        model_path = os.path.join(model_dir, model_files[selected_model])
        model = joblib.load(model_path)
        pred = model.predict(df_scaled)[0]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df_scaled)[0]
            img_base64 = plot_prediction_proba(proba, target_names)

        result = iris.target_names[pred]

    return render_template("index3.html",
                           result=result,
                           model_name=selected_model,
                           img_base64=img_base64,
                           feature_fields=zip(FEATURE_NAMES, DISPLAY_NAMES),
                           model_names=model_files.keys(),
                           prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
