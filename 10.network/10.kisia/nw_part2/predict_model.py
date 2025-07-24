# 추론 시 사용 예시
import joblib
import pandas as pd

# 로드
model = joblib.load("models/kmeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/features.pkl")
abnormal_cluster = joblib.load("models/abnormal_cluster.pkl")

# 입력 예시
sample = pd.DataFrame([{
    'method_cnt': 3,
    'method_post': 0.5,
    'protocol_1_0': 1,
    'status_major': 0.8,
    'status_404': 0,
    'status_499': 0,
    'status_cnt': 3,
    'path_same': 1,
    'path_xmlrpc': 0,
    'ua_cnt': 1,
    'has_payload': 1,
    'req_cnt_per_hour': 45
}])

# 전처리 및 예측
X_scaled = scaler.transform(sample[features])
cluster = model.predict(X_scaled)[0]
print("예측된 클러스터:", cluster)
print("이상 여부:", cluster == abnormal_cluster)
