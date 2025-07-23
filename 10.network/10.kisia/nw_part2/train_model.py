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

# 모델 저장
joblib.dump(model, "models/kmeans_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
