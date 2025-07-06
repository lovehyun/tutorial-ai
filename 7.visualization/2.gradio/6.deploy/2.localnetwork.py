# 2) Local Network 배포
#  - 같은 네트워크 내 다른 PC/모바일에서 접속 가능
#  - 서버 IP를 직접 사용

import gradio as gr

def greet(name):
    return f"안녕하세요, {name}님!"

app = gr.Interface(fn=greet, inputs="text", outputs="text")

app.launch(server_name="0.0.0.0", server_port=8080)
