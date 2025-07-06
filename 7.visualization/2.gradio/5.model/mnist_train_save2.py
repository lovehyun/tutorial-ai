# mnist_train_save.py - 개선된 버전
# 1. 모델 복잡도 부족:
#  - 기존: 128개 뉴런 1개 층
#  - 개선: 512→256개 뉴런 2개 층 + Dropout
# 2. 학습 부족:
#  - 기존: 5 에포크
#  - 개선: 15 에포크 + 조기 종료
# 3. 배치 크기 조정:
#  - 기존: 32
#  - 개선: 128 (더 안정적인 학습)
# 4. 성능 모니터링 추가:
#  - 각 숫자별 정확도 출력
#  - 학습 과정 모니터링

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np

# MNIST 데이터 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 정규화 (0~1 범위로)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 원-핫 인코딩
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 데이터 분포 확인
print("훈련 데이터 형태:", x_train.shape)
print("레이블 분포:", np.bincount(np.argmax(y_train, axis=1)))

# 개선된 모델 생성
# - 더 깊은 완전연결 신경망
# - 2개의 은닉층 (512개 → 256개 뉴런)
# - Dropout 정규화 (20% 비율)
# - 활성화 함수: ReLU → ReLU → Softmax
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

# 모델 컴파일 (학습률 조정)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 구조 출력
model.summary()

# 콜백 설정 (조기 종료)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', 
    patience=3, 
    restore_best_weights=True
)

# 모델 학습 (에포크 증가, 배치 사이즈 조정)
history = model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=128,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)

# 최종 평가
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n최종 테스트 정확도: {test_acc:.4f}")

# 각 숫자별 예측 성능 확인
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

print("\n각 숫자별 정확도:")
for i in range(10):
    mask = true_classes == i
    if np.sum(mask) > 0:
        acc = np.mean(predicted_classes[mask] == i)
        print(f"숫자 {i}: {acc:.3f}")

# 모델 저장
model.save('mnist_model.keras')

print("모델 저장 완료: mnist_model.keras")

# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  flatten (Flatten)           (None, 784)               0         
#  dense (Dense)               (None, 512)               401920    
#  dropout (Dropout)           (None, 512)               0         
#  dense_1 (Dense)             (None, 256)               131328    
#  dropout_1 (Dropout)         (None, 256)               0         
#  dense_2 (Dense)             (None, 10)                2570      
# =================================================================
# Total params: 535,818

# 1. Flatten 레이어: 0개
# 2. Dense(512) 레이어:
#  - 입력: 784 → 출력: 512
#  - 파라미터: (784 × 512) + 512 = 401,408 + 512 = 401,920
# 3. Dropout 레이어: 0개 (학습 가능한 파라미터 없음)
# 4. Dense(256) 레이어:
#  - 입력: 512 → 출력: 256
#  - 파라미터: (512 × 256) + 256 = 131,072 + 256 = 131,328
# 5. Dropout 레이어: 0개
# 6. Dense(10) 레이어:
#  - 입력: 256 → 출력: 10
#  - 파라미터: (256 × 10) + 10 = 2,560 + 10 = 2,570
#  - 전체 파라미터: 401,920 + 131,328 + 2,570 = 535,818
