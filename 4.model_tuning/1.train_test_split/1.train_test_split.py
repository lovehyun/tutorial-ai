from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 데이터 로드
X, y = load_iris(return_X_y=True)

# 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Train size:", len(X_train))
print("Test size :", len(X_test))
