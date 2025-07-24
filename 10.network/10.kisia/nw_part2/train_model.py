# train_model.py
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# 데이터 불러오기
df_entity = pd.read_csv("train_processed_2nd.csv", index_col=0)

# 사용 feature
cols_to_train = ['method_cnt','method_post','protocol_1_0','status_major','status_404','status_499',
                 'status_cnt','path_same','path_xmlrpc','ua_cnt','has_payload','req_cnt_per_hour']

# 정규화
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df_entity[cols_to_train])

# 모델 학습
model = KMeans(n_clusters=4, random_state=42)
model.fit(X_scaled)

df_entity['cluster_kmeans'] = model.labels_
cluster_counts = df_entity['cluster_kmeans'].value_counts()
print(cluster_counts)  # 각 클러스터별 개수 출력

# 예: 가장 적은 샘플 수의 클러스터를 이상으로 간주
abnormal_cluster = cluster_counts.idxmin()
joblib.dump(abnormal_cluster, "models/abnormal_cluster.pkl")

# 모델 저장
joblib.dump(model, "models/kmeans_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
