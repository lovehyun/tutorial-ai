import gradio as gr

# Gradio란?
# - Python 코드로 만든 함수에 웹 UI를 자동으로 생성해 주는 라이브러리입니다.
# 구성 요소:
# - 함수 (Function) : 입력을 받아서 출력을 생성
# - inputs : 사용자가 입력하는 UI (텍스트, 슬라이더, 이미지 등)
# - outputs : 함수 결과를 보여주는 UI (텍스트, 이미지 등)
#
# def 함수명(입력값):
#     # 처리 로직
#     return 출력값
#
# gr.Interface(fn=함수명, inputs=입력 컴포넌트, outputs=출력 컴포넌트).launch()

# 함수 정의
def greet(name):
    return f"안녕하세요, {name}님! 반갑습니다 😊"

# Gradio 인터페이스 생성
app = gr.Interface(
    fn=greet,             # 실행할 함수
    inputs=gr.Textbox(label="이름을 입력하세요"),  # 사용자 입력 UI
    outputs=gr.Textbox(label="인사 메시지")       # 출력 UI
)

# 앱 실행
app.launch()
