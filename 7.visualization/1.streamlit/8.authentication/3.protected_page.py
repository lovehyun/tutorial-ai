import streamlit as st

st.title("보호된 페이지 (로그인 필요)")

# 인증 세션 확인
if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    st.warning("이 페이지는 로그인한 사용자만 접근할 수 있습니다.")
    st.stop()

st.success(f"{st.session_state['username']}님, 보호된 페이지에 접속하셨습니다!")
