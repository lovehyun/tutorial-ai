from flask import Flask, request, render_template
import joblib, os, io, base64
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

app = Flask(__name__)

# 데이터 및 모델 정의
iris = load_iris()
X_data = pd.DataFrame(iris.data, columns=iris.feature_names)
y_data = iris.target
target_names = iris.target_names.tolist()

FEATURE_NAMES = iris.feature_names
DISPLAY_NAMES = ['꽃받침 길이', '꽃받침 너비', '꽃잎 길이', '꽃잎 너비']

model_dir = os.path.join(os.path.dirname(__file__), 'models')
model_files = {
    'RandomForest': 'rf_model.pkl',
    'SVM': 'svm_model.pkl',
    'KNN': 'knn_model.pkl',
    'NaiveBayes': 'nb_model.pkl'
}
scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))

# 예측 확률 그래프

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

# PCA 시각화 그래프

def plot_pca_with_point(user_point, user_label):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_data)
    user_pca = pca.transform([user_point])

    fig, ax = plt.subplots()
    for i, name in enumerate(target_names):
        ax.scatter(X_pca[y_data == i, 0], X_pca[y_data == i, 1], label=name, alpha=0.5)
    ax.scatter(user_pca[0, 0], user_pca[0, 1], color='red', marker='X', s=100, label='입력값')
    ax.set_title("PCA 시각화")
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    result, img_base64, pca_base64 = None, None, None
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

        pca_base64 = plot_pca_with_point(data, pred)
        result = iris.target_names[pred]

    return render_template("index4.html",
                           result=result,
                           model_name=selected_model,
                           img_base64=img_base64,
                           pca_base64=pca_base64,
                           feature_fields=zip(FEATURE_NAMES, DISPLAY_NAMES),
                           model_names=model_files.keys(),
                           prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
