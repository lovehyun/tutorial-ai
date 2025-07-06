# Gradio 주요 컴포넌트
# 컴포넌트	설명
# Textbox	텍스트 입력
# Number	숫자 입력
# Slider	슬라이더 (숫자 선택)
# Checkbox	체크박스 (참/거짓)
# Image	이미지 업로드
# Audio	오디오 업로드
# Dropdown	드롭다운 선택지

import gradio as gr

def summary(name, age, likes_python):
    python_text = "좋아합니다" if likes_python else "좋아하지 않습니다"
    return f"{name}님은 {age}살이고, Python을 {python_text}."

app = gr.Interface(
    fn=summary,
    inputs=[
        gr.Textbox(label="이름을 입력하세요"),
        gr.Slider(0, 100, label="나이 선택"),  # 0~100 범위 슬라이더
        gr.Checkbox(label="Python을 좋아하시나요?")  # 체크박스
    ],
    outputs=gr.Textbox(label="요약 결과")
)

app.launch()
