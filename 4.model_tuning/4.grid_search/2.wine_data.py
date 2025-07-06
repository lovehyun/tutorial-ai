from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. 데이터 로드 및 분할
wine = load_wine()
X = wine.data
y = wine.target

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
print(classification_report(y_test, y_pred, target_names=wine.target_names))


# 핵심 설명
# ✔️ wine 데이터는 iris보다 변수 개수가 많아서 파라미터 튜닝의 효과가 더 크게 나타납니다.
# ✔️ 이 코드는 iris 실습과 완전히 동일한 구조이며, 데이터셋만 wine으로 변경한 것입니다.
# ✔️ cv=5로 5-Fold 교차 검증을 자동으로 반복해주고,
# ✔️ f1_macro를 기준으로 가장 좋은 모델을 찾아줍니다.
