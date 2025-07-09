# 2_feature_engineering.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# CSV 데이터 로드
df = pd.read_csv("network_multiclass.csv")

print("데이터 로드 완료")
print(df.head())

# ----------------------------
# 1. 클래스 분포 시각화
# ----------------------------
sns.countplot(data=df, x='label')
plt.title("클래스 분포 (0=Normal, 1=DoS, 2=Probe)")
plt.show()

# ----------------------------
# 2. 상관관계 히트맵
# ----------------------------
corr = df.drop(columns='label').corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("특성 간 상관관계")
plt.show()

# ----------------------------
# 3. 표준화 및 정규화
# ----------------------------
features = df.drop(columns='label')
labels = df['label']

# 선택 1: 표준화 (평균 0, 분산 1)
scaler_std = StandardScaler()
features_std = scaler_std.fit_transform(features)

# 선택 2: 정규화 (0~1 스케일)
scaler_minmax = MinMaxScaler()
features_norm = scaler_minmax.fit_transform(features)

print("표준화 및 정규화 완료")

# ----------------------------
# 4. PCA 시각화 (2D 투영)
# ----------------------------
pca = PCA(n_components=2)
components = pca.fit_transform(features_std)

df_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
df_pca['label'] = labels

sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='label', palette='deep')
plt.title("PCA 2D 시각화 (Standard Scaled)")
plt.show()
