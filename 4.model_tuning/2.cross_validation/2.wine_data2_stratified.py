# 핵심 포인트
# 항목	설명
# K-Fold	데이터를 K조각으로 나눠서 순서대로 검증
# shuffle=True	데이터 섞기 (필수)
# Stratified K-Fold	라벨 비율을 반드시 유지 (필수)

# ✔️ wine 데이터처럼 다중 클래스 분류 문제는 반드시 Stratified K-Fold를 사용해야 합니다.

from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedKFold

# 데이터 로드
wine = load_wine()
X = wine.data
y = wine.target

# Stratified K-Fold 준비 (라벨 비율 유지)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold = 1
for train_index, test_index in skf.split(X, y):
    print(f"Fold {fold}")
    print(f"훈련 데이터 개수: {len(train_index)}, 검증 데이터 개수: {len(test_index)}")
    print(f"훈련 데이터 클래스 비율: {y[train_index].tolist().count(0)}/{y[train_index].tolist().count(1)}/{y[train_index].tolist().count(2)}")
    print(f"검증 데이터 클래스 비율: {y[test_index].tolist().count(0)}/{y[test_index].tolist().count(1)}/{y[test_index].tolist().count(2)}\n")
    fold += 1
