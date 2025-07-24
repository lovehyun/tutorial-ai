import pandas as pd
import joblib

data = joblib.load("models/kmeans_model.pkl")

print("타입:", type(data))
print("내용 일부:", str(data)[:300])  # 너무 길면 앞부분만

# 예: 피처 이름이 들어 있는 dict 또는 DataFrame일 때
if isinstance(data, dict) and "features" in data:
    print("피처 목록:", data["features"])

elif isinstance(data, pd.DataFrame):
    print("피처 목록:", data.columns.tolist())

# 혹시 KMeans 모델이라면?
if hasattr(data, 'cluster_centers_'):
    print("KMeans 모델로 추정됨")
