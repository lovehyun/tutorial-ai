import streamlit as st
import requests

st.title("FastAPI 연동 (동기 요청)")

name = st.text_input("이름을 입력하세요:")

if st.button("FastAPI 호출"):
    url = f"http://localhost:8000/api/greet/{name}"
    response = requests.get(url)

    if response.status_code == 200:
        st.write(response.json()['message'])
    else:
        st.write("서버 에러가 발생했습니다.")
