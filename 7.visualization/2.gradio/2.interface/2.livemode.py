import gradio as gr

def double(number):
    return number * 2

app = gr.Interface(
    fn=double,
    inputs=gr.Number(label="숫자 입력"),
    outputs=gr.Number(label="두 배 결과"),
    live=True  # 입력이 바뀌면 바로 결과 업데이트
)

app.launch()
