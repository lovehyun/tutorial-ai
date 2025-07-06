import streamlit as st
import pandas as pd
import numpy as np

st.title("라인 차트 그리기")

# 샘플 데이터 생성
data = pd.DataFrame({
    'x': np.arange(0, 100),
    'y': np.random.randn(100).cumsum()
})

st.line_chart(data.set_index('x'))
