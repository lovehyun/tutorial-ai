import gradio as gr

def double(x):
    return x * 2

with gr.Blocks() as demo:
    num_input = gr.Number(label="숫자 입력")
    output = gr.Textbox()

    num_input.change(fn=double, inputs=num_input, outputs=output)

demo.launch()
