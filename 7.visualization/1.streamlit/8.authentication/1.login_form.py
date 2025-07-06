import streamlit as st

st.title("로그인 페이지")

# 사용자 데이터 (임시 DB)
db_users = {"user": "password123"}

username = st.text_input("사용자 이름")
password = st.text_input("비밀번호", type="password")

if st.button("로그인"):
    if username in db_users and db_users[username] == password:
        st.session_state["authenticated"] = True
        st.session_state["username"] = username
        st.success(f"{username}님, 로그인 성공!")
    else:
        st.error("로그인 실패: 사용자 이름 또는 비밀번호가 틀렸습니다.")
