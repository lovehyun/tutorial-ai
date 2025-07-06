from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# 1. 데이터 로드
wine = load_wine()
X = wine.data
y = wine.target

# 2. 데이터 분할 (80% 훈련, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"훈련 데이터 개수: {len(X_train)}")
print(f"테스트 데이터 개수: {len(X_test)}")

print(f"훈련 데이터 클래스 비율: {y_train.tolist().count(0)}/{y_train.tolist().count(1)}/{y_train.tolist().count(2)}")
print(f"테스트 데이터 클래스 비율: {y_test.tolist().count(0)}/{y_test.tolist().count(1)}/{y_test.tolist().count(2)}")
