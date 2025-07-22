# train_and_export.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import joblib
import os

# 데이터 로드 및 전처리
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 모델들 정의
models = {
    "rf_model.pkl": RandomForestClassifier(),
    "svm_model.pkl": SVC(probability=True),
    "knn_model.pkl": KNeighborsClassifier(),
    "nb_model.pkl": GaussianNB()
}

# 학습 및 저장 디렉토리
os.makedirs("models", exist_ok=True)
for filename, model in models.items():
    model.fit(X_scaled, y)
    joblib.dump(model, f"models/{filename}")

# 스케일러 저장
joblib.dump(scaler, "models/scaler.pkl")
print("모델 및 스케일러 저장 완료!")
