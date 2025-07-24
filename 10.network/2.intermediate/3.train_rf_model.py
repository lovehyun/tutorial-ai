# train_rf_model.py
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 데이터 로드
df = pd.read_csv("network_multiclass.csv")
X = df.drop(columns="label")
y = df["label"]

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 모델 정의 및 학습 (스케일러 포함)
rf_model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
rf_model.fit(X_train, y_train)

# 평가
y_pred = rf_model.predict(X_test)
print("Random Forest 결과")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 모델 저장
os.makedirs("models", exist_ok=True)
joblib.dump(rf_model, "models/rf_model.pkl")
print("Random Forest 모델 저장 완료 → models/rf_model.pkl")
