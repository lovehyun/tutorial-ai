import gradio as gr

def greet(name):
    return f"반갑습니다, {name}님!"

with gr.Blocks() as demo:
    name_input = gr.Textbox(label="이름 입력", lines=1)
    output = gr.Textbox()

    name_input.submit(fn=greet, inputs=name_input, outputs=output)

demo.launch()
