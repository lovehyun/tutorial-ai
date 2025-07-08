# analysis.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import io
from PIL import Image

import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 기준
matplotlib.rcParams['axes.unicode_minus'] = False     # 음수 부호 깨짐 방지

uploaded_df = None  # 전역 변수

def load_csv(file):
    global uploaded_df
    if file is None:
        return "CSV 파일을 업로드해주세요.", None
    try:
        uploaded_df = pd.read_csv(file.name)
        # return "파일 로그 성공!", uploaded_df.head()
        return "파일 로그 성공!", uploaded_df
    except Exception as e:
        return f"파일 로드 실패: {e}", None

def plot_sales():
    if uploaded_df is None:
        return None
    df = uploaded_df.copy()
    if df.shape[1] < 2:
        return None

    x = df.iloc[:, 0]
    y = df.iloc[:, 1]

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, marker='o', label='월별 판매량')
    plt.xticks(rotation=45)
    plt.title("월별 판매량")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img = Image.open(buf)  # PIL Image 객체로 변환
    return img

def analyze_trend():
    if uploaded_df is None:
        return None, "파일이 업로드되지 않았습니다."

    df = uploaded_df.copy()
    if df.shape[1] < 2:
        return None, "2개 이상의 컬럼이 필요합니다."

    x = np.arange(len(df)).reshape(-1, 1)
    y = df.iloc[:, 1].values

    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)

    next_1 = model.predict([[len(df)]])[0]
    next_2 = model.predict([[len(df) + 1]])[0]

    plt.figure(figsize=(8, 4))
    plt.plot(df.iloc[:, 0], y, marker='o', label='실제 판매량')
    plt.plot(df.iloc[:, 0], y_pred, linestyle='--', label='추세선')

    future_labels = ["다음달", "다다음달"]
    future_x = np.array([len(df), len(df) + 1]).reshape(-1, 1)
    future_y = model.predict(future_x)

    full_labels = df.iloc[:, 0].tolist() + future_labels
    full_y = np.concatenate([y, future_y])

    plt.plot(future_labels, future_y, 'ro--', label='예측값')
    plt.xticks(rotation=45)
    plt.title("판매량 추세 분석")
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    img = Image.open(buf)  # PIL 이미지로 변환해야 함

    result = f"다음달 예상 판매량: {next_1:.1f}\n다다음달 예상 판매량: {next_2:.1f}"
    return img, result
