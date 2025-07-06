# ✅ 핵심 포인트
# shuffle=True는 반드시 써야 데이터를 잘 섞을 수 있어요.
# stratify 기능이 자동으로 적용돼서 각 Fold별 클래스 비율이 유지됩니다.

# ✅ 결론
# ✔️ iris도 반드시 Stratified K-Fold를 써야 합니다.
# ✔️ 데이터셋이 작고, 클래스가 3개인 경우는 분할 시 라벨 비율이 쉽게 깨질 수 있기 때문이에요.

from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold

# 1. 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

# 2. Stratified K-Fold 준비 (5개 조각)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold = 1
for train_index, test_index in skf.split(X, y):
    print(f"Fold {fold}")
    print(f"훈련 데이터 개수: {len(train_index)}, 검증 데이터 개수: {len(test_index)}")

    # 훈련 데이터 클래스 비율 출력
    print(f"훈련 데이터 클래스 비율: "
          f"{y[train_index].tolist().count(0)}/"
          f"{y[train_index].tolist().count(1)}/"
          f"{y[train_index].tolist().count(2)}")

    # 검증 데이터 클래스 비율 출력
    print(f"검증 데이터 클래스 비율: "
          f"{y[test_index].tolist().count(0)}/"
          f"{y[test_index].tolist().count(1)}/"
          f"{y[test_index].tolist().count(2)}\n")

    fold += 1
