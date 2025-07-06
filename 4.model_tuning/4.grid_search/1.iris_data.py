# 지도학습 4단계: GridSearchCV (모델 튜닝)
# ✔️ 핵심 개념
# GridSearchCV는 다양한 파라미터 조합을 모두 시도해보고
# 👉 가장 성능이 좋은 조합을 자동으로 찾아주는 도구입니다.
# 교차 검증을 자동으로 함께 진행해줍니다. (K-Fold 포함)

# 핵심 파라미터 설명
# 파라미터	설명
# param_grid	실험할 파라미터 조합
# cv=5	5-Fold 교차 검증
# scoring='f1_macro'	다중 클래스에서 F1-Score를 최적화
# n_jobs=-1	모든 CPU 코어 사용 (속도 향상)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. 데이터 로드 및 분할
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. 하이퍼파라미터 후보군 설정
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [2, 4, 6],
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
print(classification_report(y_test, y_pred, target_names=iris.target_names))
