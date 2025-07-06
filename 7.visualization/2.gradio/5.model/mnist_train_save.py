# mnist_train_save.py
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# MNIST 데이터 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # 정규화

# 원-핫 인코딩
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 모델 생성
# - 간단한 완전연결 신경망 (Fully Connected Neural Network)
# - 1개의 은닉층 (128개 뉴런)
# - 활성화 함수: ReLU → Softmax
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# 모델 저장 (Keras v3 포맷 권장)
model.save('mnist_model.keras')

print("모델 저장 완료: mnist_model.keras")

# mnist_model.keras:
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  flatten (Flatten)           (None, 784)               0         
#  dense (Dense)               (None, 128)               100480    
#  dense_1 (Dense)             (None, 10)                1290      
# =================================================================
# Total params: 101,770

# 파라미터 수 = (입력 뉴런 수 × 출력 뉴런 수) + 출력 뉴런 수
#              |_________가중치_________|   |__편향__|
#
# 1. Flatten 레이어:
#  - 입력: (28, 28) → 출력: (784,)
#  - 파라미터: 0개 (단순 형태 변환)
# 2. Dense(128) 레이어:
#  - 입력: 784 → 출력: 128
#  - 가중치: 784 × 128 = 100,352
#  - 편향: 128
#  - 총 파라미터: 100,352 + 128 = 100,480
# 3. Dense(10) 레이어:
#  - 입력: 128 → 출력: 10
#  - 가중치: 128 × 10 = 1,280
#  - 편향: 10
#  - 총 파라미터: 1,280 + 10 = 1,290
#  - 전체 파라미터: 100,480 + 1,290 = 101,770
