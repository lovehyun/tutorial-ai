import gradio as gr

# 상태 저장용 챗봇
def chatbot(user_input, history):
    history = history or []
    history.append((user_input, f"답변: {user_input}"))
    return history, history

# 계산기
def calculator(a, b, operation):
    if operation == "더하기":
        return a + b
    elif operation == "빼기":
        return a - b
    elif operation == "곱하기":
        return a * b
    elif operation == "나누기":
        return a / b if b != 0 else "0으로 나눌 수 없습니다."

# 이미지 분류기 (간단 예시)
def classify_image(image):
    return "이 이미지는 샘플 이미지입니다."

# 탭 1: 챗봇
with gr.Blocks() as chatbot_tab:
    gr.Markdown("## 챗봇 (상태 유지)")
    with gr.Row():
        chatbot_state = gr.State([])
        chat_input = gr.Textbox(label="질문 입력")
        chat_output = gr.Chatbot(label="대화 기록")
    chat_input.submit(chatbot, [chat_input, chatbot_state], [chat_output, chatbot_state])

# 탭 2: 계산기
with gr.Blocks() as calculator_tab:
    gr.Markdown("## 계산기")
    with gr.Row():
        num1 = gr.Number(label="숫자 1")
        num2 = gr.Number(label="숫자 2")
        operation = gr.Dropdown(["더하기", "빼기", "곱하기", "나누기"], label="연산 선택")
    calc_button = gr.Button("계산하기")
    calc_output = gr.Textbox(label="결과")
    calc_button.click(calculator, [num1, num2, operation], calc_output)

# 탭 3: 이미지 분류기
with gr.Blocks() as image_tab:
    gr.Markdown("## 이미지 분류기")
    image_input = gr.Image(label="이미지 업로드")
    image_output = gr.Textbox(label="분류 결과")
    image_input.change(classify_image, image_input, image_output)

# 전체 앱: 탭 구성
with gr.Blocks() as app:
    gr.TabbedInterface([chatbot_tab, calculator_tab, image_tab], ["챗봇", "계산기", "이미지 분류기"])

app.launch()
