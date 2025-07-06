import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# 모델 로드
model = tf.keras.models.load_model('mnist_model.keras')

def predict_digit(image):
    """MNIST 숫자 예측 함수"""
    if image is None:
        return "이미지를 그려주세요"
    
    # Gradio 딕셔너리에서 이미지 추출
    if isinstance(image, dict):
        image = image.get('composite')
        if image is None:
            return "이미지를 그려주세요"
    
    # numpy 배열을 PIL로 변환해서 28x28 그레이스케일로 변환
    image = Image.fromarray(image).convert('L').resize((28, 28))
    image = np.array(image)
    
    # MNIST 형식에 맞게 색상 반전 (흰 배경/검은 글씨 → 검은 배경/흰 글씨)
    image = 255 - image
    
    # 정규화
    image = image / 255.0
    
    # 모델 입력 형태로 변환 (1, 28, 28)
    image = image.reshape(1, 28, 28)
    
    # 예측
    predictions = model.predict(image, verbose=0)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    
    return f"예측: {predicted_class} (신뢰도: {confidence:.3f})"

# Gradio 인터페이스 생성 및 실행
app = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(),
    outputs=gr.Textbox(label="예측 결과"),
    live=True
)

app.launch()
