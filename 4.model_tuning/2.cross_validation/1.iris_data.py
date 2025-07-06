from sklearn.datasets import load_iris
from sklearn.model_selection import KFold

# 1. 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

# 2. K-Fold 준비 (5개 조각으로 나누기)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 3. Fold 별로 인덱스 출력
fold = 1
for train_index, test_index in kf.split(X):
    print(f"Fold {fold}")
    print(f"훈련 데이터: {train_index}")
    print(f"검증 데이터: {test_index}\n")
    fold += 1
