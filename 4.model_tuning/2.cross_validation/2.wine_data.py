from sklearn.datasets import load_wine
from sklearn.model_selection import KFold

# 데이터 로드
wine = load_wine()
X = wine.data
y = wine.target

# K-Fold 준비 (5개 조각으로 나누기)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold = 1
for train_index, test_index in kf.split(X):
    print(f"Fold {fold}")
    print(f"훈련 데이터 개수: {len(train_index)}, 검증 데이터 개수: {len(test_index)}")
    fold += 1
