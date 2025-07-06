import gradio as gr

def square(number):
    return number ** 2

app = gr.Interface(
    fn=square,
    inputs=gr.Number(label="숫자를 입력하세요"),
    outputs=gr.Number(label="제곱 결과")
)

app.launch()
