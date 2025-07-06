from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. 데이터 로드 및 분할
digits = load_digits()
X = digits.data
y = digits.target

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
print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))


# 핵심 포인트
# ✔️ digits 데이터는 클래스가 10개이기 때문에, f1_macro를 반드시 사용하는 것이 중요합니다.
# ✔️ classification_report에서 클래스별 Precision, Recall, F1-Score를 모두 확인할 수 있습니다.
# ✔️ stratify=y는 반드시 유지해야 클래스 비율이 훈련/테스트에 동일하게 유지됩니다.
