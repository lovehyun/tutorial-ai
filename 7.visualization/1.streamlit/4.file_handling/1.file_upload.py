import streamlit as st

st.title("파일 업로드 (텍스트 파일)")

uploaded_file = st.file_uploader("텍스트 파일을 업로드하세요.", type=["txt"])

if uploaded_file is not None:
    content = uploaded_file.read().decode('utf-8')
    st.write("파일 내용:")
    st.write(content)
