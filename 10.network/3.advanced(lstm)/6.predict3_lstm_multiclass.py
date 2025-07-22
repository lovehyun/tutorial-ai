from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. 모델 불러오기
model = load_model("model_lstm_multiclass_seq10.h5")

# 2. 데이터 로드
data = np.load("lstm_data.npz")
X = data["X"]
y = data["y"].astype(int)  # 정수형 레이블 (0, 1, 2)

# 3. One-hot 인코딩
y_cat = to_categorical(y, num_classes=3)

# 4. 데이터 분할
_, X_test, _, y_test = train_test_split(X, y_cat, test_size=0.2, stratify=y, random_state=42)

# 5. 예측 및 평가
y_pred = model.predict(X_test).argmax(axis=1)
y_true = y_test.argmax(axis=1)

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
