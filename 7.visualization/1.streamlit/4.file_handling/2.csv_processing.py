import streamlit as st
import pandas as pd

st.title("CSV 파일 업로드 및 분석")

uploaded_file = st.file_uploader("CSV 파일을 업로드하세요.", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("업로드한 데이터프레임:")
    st.dataframe(df)

    st.write("간단한 통계 요약:")
    st.write(df.describe())
