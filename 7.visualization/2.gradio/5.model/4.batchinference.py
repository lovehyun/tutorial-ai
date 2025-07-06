# Hugging Face 사전학습 모델 연동
# 결과: 여러 문장을 한 번에 처리해 감정을 분석할 수 있어요.

import gradio as gr
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

def batch_sentiment(texts):
    results = classifier(texts)
    return [f"{text}: {res['label']} ({res['score']:.2f})" for text, res in zip(texts, results)]

app = gr.Interface(
    fn=batch_sentiment,
    inputs=gr.Textbox(lines=5, label="여러 문장을 줄바꿈으로 입력하세요"),
    outputs=gr.Textbox(label="배치 결과"),
    examples=[["I love this!", "This is terrible.", "It's okay."]]
)

app.launch()
