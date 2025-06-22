from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

X, y = load_iris(return_X_y=True)

# 파라미터 후보
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

# GridSearchCV 적용
grid = GridSearchCV(SVC(), param_grid, cv=3)
grid.fit(X, y)

print("Best parameters:", grid.best_params_)
print("Best score:", round(grid.best_score_, 3))
