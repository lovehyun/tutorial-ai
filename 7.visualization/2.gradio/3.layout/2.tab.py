import gradio as gr

def greet(name):
    return f"안녕하세요, {name}님!"

def square(num):
    return num ** 2

with gr.Blocks() as demo:
    gr.Markdown("# 탭 레이아웃 예제")

    with gr.Tab("인사하기"):
        name_input = gr.Textbox(label="이름 입력")
        greet_button = gr.Button("인사하기")
        greet_output = gr.Textbox(label="결과")
        greet_button.click(greet, inputs=name_input, outputs=greet_output)

    with gr.Tab("제곱 계산기"):
        num_input = gr.Number(label="숫자 입력")
        square_button = gr.Button("제곱 계산")
        square_output = gr.Textbox(label="결과")
        square_button.click(square, inputs=num_input, outputs=square_output)

demo.launch()
