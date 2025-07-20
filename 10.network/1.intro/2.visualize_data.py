# 2_visualize_data.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import platform

# 한글 폰트 설정
if platform.system() == 'Windows':
    matplotlib.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':  # macOS
    matplotlib.rc('font', family='AppleGothic')
else:  # Linux (NanumGothic 설치 필요)
    matplotlib.rc('font', family='NanumGothic')

# 마이너스 깨짐 방지
matplotlib.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("network_data.csv")

print("데이터 요약:")
print(df.describe())

# 1. 클래스 분포
sns.countplot(data=df, x='label')
plt.title("정상(0) / 이상(1) 트래픽 수")
plt.show()

# 2. 상관관계 히트맵
corr = df.drop(columns='label').corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("특성 간 상관관계")
plt.show()

# 3. duration 분포
sns.histplot(data=df, x="duration", hue="label", kde=True, bins=30)
plt.title("Duration 분포 (정상 vs 이상)")
plt.show()
