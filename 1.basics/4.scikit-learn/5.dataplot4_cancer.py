import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# 데이터 로드
cancer = load_breast_cancer()
cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
cancer_df['target'] = cancer.target

# 타겟 분포 카운트플롯
plt.figure()
sns.countplot(x='target', data=cancer_df)
plt.title('Breast Cancer Target Distribution (0 = Malignant, 1 = Benign)')
plt.show()

# 산점도 (mean radius vs mean texture)
plt.figure()
sns.scatterplot(x='mean radius', y='mean texture', hue='target', data=cancer_df)
plt.title('Mean Radius vs Mean Texture')
plt.show()

# 박스플롯
plt.figure()
sns.boxplot(x='target', y='mean area', data=cancer_df)
plt.title('Mean Area by Class')
plt.show()

# 바이올린 플롯
plt.figure()
sns.violinplot(x='target', y='mean smoothness', data=cancer_df)
plt.title('Mean Smoothness by Class')
plt.show()

# 히스토그램
plt.figure()
sns.histplot(cancer_df['mean perimeter'], bins=30, kde=True)
plt.title('Mean Perimeter Distribution')
plt.show()

# 상관계수 히트맵
plt.figure(figsize=(12, 10))
sns.heatmap(cancer_df.corr(), cmap='coolwarm')
plt.title('Breast Cancer Feature Correlation')
plt.show()

# 페어플롯 (상위 4개 특성)
top4 = cancer_df.corr()['target'].abs().sort_values(ascending=False).index[1:5]
sns.pairplot(cancer_df[top4.to_list() + ['target']], hue='target')
plt.show()
