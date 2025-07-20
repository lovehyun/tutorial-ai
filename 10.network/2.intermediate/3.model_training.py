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

# XGBoost 결과
# [[78  0  2]
#  [ 0 80  0]
#  [ 0  0 80]]
# 클래스 0에서 2개 샘플이 클래스 2로 잘못 분류됨

# 오분류 샘플 보기
# misclassified_xgb = X_test[y_test != xgb_preds]
# print("XGBoost 오분류 샘플:")
# print(misclassified_xgb)

# 오분류 샘플 및 예측 결과
# mismatch = X_test[y_test != xgb_preds].copy()
# mismatch['true_label'] = y_test[y_test != xgb_preds]
# mismatch['predicted_label'] = xgb_preds[y_test != xgb_preds]
# print(mismatch)

#       duration  packet_size   src_bytes   dst_bytes  true_label  predicted_label
# 273  13.444771   185.333744  607.478727  595.487808           0                2
# 263  19.375904   220.981864  658.548739  609.474982           0                2
