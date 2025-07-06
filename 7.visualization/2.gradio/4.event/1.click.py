# Gradio 주요 이벤트 종류
# 이벤트	설명
# click	버튼 클릭 시 실행
# change	값이 바뀔 때 자동 실행
# submit	Enter 또는 제출 시 실행
# upload	파일 업로드 시 실행
# async	비동기 함수 지원 (대기 시간 처리 가능)

import gradio as gr

def say_hello():
    return "안녕하세요!"

with gr.Blocks() as demo:
    btn = gr.Button("인사하기")
    output = gr.Textbox()

    btn.click(fn=say_hello, inputs=None, outputs=output)

demo.launch()
