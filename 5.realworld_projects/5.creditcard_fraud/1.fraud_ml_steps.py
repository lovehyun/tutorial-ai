# Credit Card Fraud Detection 단계별 머신러닝 학습 코드

# =========================
# 1-1. 데이터 로드 및 기본 정보 확인
# =========================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

# 데이터 로드
df = pd.read_csv('creditcard.csv')

# 데이터 구조 확인
print('# 1-1 데이터 구조')
print(df.shape)
print(df.info())
print(df.head())

# 클래스 분포 확인
print('# 1-1 클래스 분포')
print(df['Class'].value_counts())


# =========================
# 1-2. 클래스 분포 시각화
# =========================
counts = df['Class'].value_counts()
labels = ['Normal', 'Fraud']

plt.figure(figsize=(6, 4))
plt.bar(labels, counts)
plt.title('Class Distribution')
plt.show()


# =========================
# 1-3. 정규화 (Amount, Time)
# =========================
scaler = StandardScaler()
df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_Scaled'] = scaler.fit_transform(df[['Time']])

# 기존 피처 삭제
df = df.drop(['Amount', 'Time'], axis=1)


# =========================
# 2-1. 데이터 분리
# =========================
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# =========================
# 2-2. 로지스틱 회귀 기본 학습
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('# 2-2 기본 로지스틱 회귀 결과')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# =========================
# 3-1. SMOTE 적용 (Oversampling)
# =========================
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print('# 3-1 SMOTE 적용 후 클래스 분포')
print(pd.Series(y_train_smote).value_counts())


# =========================
# 3-2. SMOTE 적용 후 재학습
# =========================
model_smote = LogisticRegression(max_iter=1000)
model_smote.fit(X_train_smote, y_train_smote)

y_pred_smote = model_smote.predict(X_test)

print('# 3-2 SMOTE 재학습 결과')
print(confusion_matrix(y_test, y_pred_smote))
print(classification_report(y_test, y_pred_smote))


# =========================
# 4-1. Isolation Forest 비지도 학습
# =========================
iso_model = IsolationForest(contamination=0.001, random_state=42)
iso_model.fit(X_train)

# predict: 정상 1, 이상치 -1
y_pred_iso = iso_model.predict(X_test)
y_pred_iso = [1 if x == -1 else 0 for x in y_pred_iso]  # 이상치만 1로 변환

print('# 4-1 Isolation Forest 결과')
print(confusion_matrix(y_test, y_pred_iso))
print(classification_report(y_test, y_pred_iso))


# =========================
# 5-1. 지도학습 vs 비지도학습 AUC 비교
# =========================
print('# 5-1 지도학습 AUC:', roc_auc_score(y_test, model_smote.predict_proba(X_test)[:, 1]))
print('# 5-1 비지도학습 AUC:', roc_auc_score(y_test, y_pred_iso))


# =========================
# 6-1. 모델 튜닝 (Grid Search)
# =========================
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=3, scoring='f1', n_jobs=-1)
grid.fit(X_train_smote, y_train_smote)

print('# 6-1 Grid Search Best Parameters:', grid.best_params_)
print('# 6-1 Grid Search Best F1 Score:', grid.best_score_)


# =========================
# 7-1. Stratified K-Fold Cross Validation
# =========================
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression(max_iter=1000)
scores = cross_val_score(model, X, y, cv=kfold, scoring='f1', n_jobs=-1)

print('# 7-1 K-Fold F1 Scores:', scores)
print('# 7-1 K-Fold Average F1 Score:', scores.mean())


# =========================
# 8-1. 최종 파이프라인 구성
# =========================
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print('# 8-1 최종 파이프라인 결과')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
