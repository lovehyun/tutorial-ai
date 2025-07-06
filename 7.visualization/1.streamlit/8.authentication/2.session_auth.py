import streamlit as st

st.title("세션 인증 + 페이지 보호")

# 임시 사용자 DB
db_users = {"user": "password123"}

# 세션 인증 상태 초기화
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# 로그인 기능
if not st.session_state["authenticated"]:
    username = st.text_input("사용자 이름")
    password = st.text_input("비밀번호", type="password")

    if st.button("로그인"):
        if username in db_users and db_users[username] == password:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.experimental_rerun()
        else:
            st.error("로그인 실패")

# 로그인 성공 시 화면
if st.session_state["authenticated"]:
    st.success(f"{st.session_state['username']}님, 환영합니다!")
    if st.button("로그아웃"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = ""
        st.experimental_rerun()
