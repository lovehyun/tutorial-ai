import streamlit as st
import pandas as pd
import numpy as np

st.title("바 차트 그리기")

# 샘플 데이터 생성
data = pd.DataFrame({
    '항목': ['A', 'B', 'C', 'D', 'E'],
    '값': np.random.randint(10, 100, 5)
})

st.bar_chart(data.set_index('항목'))
