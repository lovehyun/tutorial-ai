from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. 데이터 로드 및 분할
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. 하이퍼파라미터 후보군 설정
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [4, 6, 8],
    'min_samples_split': [2, 4]
}

# 3. GridSearchCV 준비
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)

# 4. 모델 학습
grid_search.fit(X_train, y_train)

# 5. 최적의 파라미터 확인
print(f"최적의 파라미터: {grid_search.best_params_}")
print(f"최고 F1-Score (교차 검증 평균): {grid_search.best_score_:.4f}")

# 6. 테스트 데이터 평가
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n=== 테스트 데이터 평가 결과 ===")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))


# 핵심 포인트
# 항목	설명
# 데이터셋	breast cancer (이진 분류, 불균형 존재)
# 평가 지표	f1_macro (이진 분류에도 유효)
# 라벨 비율 유지	반드시 stratify=y 사용

# ✔️ breast cancer 데이터는 정상:악성 비율이 2:1 정도로 불균형이 있기 때문에,
# ✔️ precision, recall, f1-score를 반드시 같이 확인해야 합니다.
# ✔️ 교차 검증 자동 수행 → 과적합 방지 + 튜닝 자동화.
