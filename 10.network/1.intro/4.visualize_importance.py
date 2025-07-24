# 4_visualize_importance.py
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# 모델과 데이터 로딩
model = joblib.load("rf_model.pkl")
df = pd.read_csv("network_data.csv")
X = df.drop(columns="label")

# 중요도 추출
importances = model.feature_importances_
features = X.columns

# 정렬
sorted_idx = importances.argsort()
features_sorted = features[sorted_idx]
importances_sorted = importances[sorted_idx]

# 시각화
plt.figure(figsize=(10, 6))
plt.barh(features_sorted, importances_sorted)
plt.title("Feature Importances (RandomForest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
