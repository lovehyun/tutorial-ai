# 1) Public Share URL (Gradio 기본 배포)
#  - Gradio가 무료 Public URL 발급 (임시 링크)
#  - 서버 종료 시 링크도 사라짐

import gradio as gr

def greet(name):
    return f"안녕하세요, {name}님!"

app = gr.Interface(fn=greet, inputs="text", outputs="text")
app.launch(share=True)  # public 링크 발급
