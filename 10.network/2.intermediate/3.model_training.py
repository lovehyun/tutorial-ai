# 3_model_training.py
# pip install scikit-learn xgboost

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# XGBoost가 설치되어 있다면 import
try:
    from xgboost import XGBClassifier
    has_xgboost = True
except ImportError:
    print("xgboost 패키지가 설치되어 있지 않아 해당 모델은 건너뜁니다.")
    has_xgboost = False

# 데이터 로드
df = pd.read_csv("network_multiclass.csv")
X = df.drop(columns="label")
y = df["label"]

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Random Forest 모델 학습
rf_model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
rf_model.fit(X_train, y_train)

# 예측 및 평가
rf_preds = rf_model.predict(X_test)

print("Random Forest 결과")
print(confusion_matrix(y_test, rf_preds))
print(classification_report(y_test, rf_preds))

# XGBoost 모델도 실행 (설치되어 있다면)
if has_xgboost:
    xgb_model = make_pipeline(StandardScaler(), XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
    xgb_model.fit(X_train, y_train)

    xgb_preds = xgb_model.predict(X_test)

    print("XGBoost 결과")
    print(confusion_matrix(y_test, xgb_preds))
    print(classification_report(y_test, xgb_preds))
