from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. 데이터 로드 및 분할
wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. 파이프라인 생성
pipe = Pipeline([
    ('scaler', StandardScaler()),  # 데이터 스케일링
    ('rf', RandomForestClassifier(random_state=42))  # 랜덤 포레스트
])

# 3. GridSearchCV 준비
param_grid = {
    'rf__n_estimators': [50, 100, 150],
    'rf__max_depth': [2, 4, 6],
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
print(classification_report(y_test, y_pred, target_names=wine.target_names))


# 핵심 정리
# ✔️ wine 데이터도 Pipeline 구조 그대로 사용
# ✔️ 파라미터 튜닝은 rf__파라미터명으로 연동
# ✔️ f1_macro를 사용하면 다중 클래스에서도 공평하게 성능을 평가할 수 있어요.
