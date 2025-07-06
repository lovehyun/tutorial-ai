import streamlit as st

st.title("이미지 분류기")

uploaded_image = st.file_uploader("이미지를 업로드하세요.", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="업로드한 이미지", use_column_width=True)
    st.write("이 이미지는 샘플 이미지입니다.")
