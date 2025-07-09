import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

# 모델 불러오기
model = load_model("model_lstm_seq10.h5")

# 데이터 로드 및 분할
data = np.load("lstm_data.npz")
X = data["X"]
y = (data["y"] != 0).astype(int)

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 추론
y_pred = (model.predict(X_test) > 0.5).astype(int)

# 평가 출력
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
