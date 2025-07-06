from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. 데이터 로드 및 분할
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. 파이프라인 생성
pipe = Pipeline([
    ('scaler', StandardScaler()),  # 데이터 표준화
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42))  # 랜덤 포레스트
])

# 3. GridSearchCV 준비
param_grid = {
    'rf__n_estimators': [50, 100, 150],
    'rf__max_depth': [4, 6, 8],
    'rf__min_samples_split': [2, 4]
}

grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)

# 4. 모델 학습
grid_search.fit(X_train, y_train)

# 5. 최적 파라미터 확인
print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최고 F1-Score (교차 검증 평균): {grid_search.best_score_:.4f}")

# 6. 테스트 데이터 평가
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n=== 테스트 데이터 평가 결과 ===")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))


# 핵심 포인트
# 단계	설명
# Pipeline	전처리 + 모델을 하나로 묶음
# GridSearchCV	파이프라인 전체를 튜닝
# rf__n_estimators	rf는 파이프라인 단계 이름, __는 GridSearchCV 문법

# ✔️ Pipeline을 사용하면 전처리 과정도 교차 검증 안에 포함됩니다.
# ✔️ GridSearchCV와 파이프라인을 함께 쓰면 코드가 매우 깔끔하고 실수 없이 완성됩니다.
