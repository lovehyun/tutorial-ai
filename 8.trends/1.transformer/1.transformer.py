# pip install torch

# -----------------------------------------------
# 이 코드는 미니 트랜스포머를 이용한 "다음 단어 예측" 모델입니다.
# 전체 흐름:
# 1. 샘플 문장 준비 → 단어장 생성 → 입력/타겟 시퀀스 변환
# 2. 미니 트랜스포머 모델 정의 (Embedding + Attention + FC)
# 3. CrossEntropyLoss로 단어 예측 학습
# 4. 학습 후, 사용자가 문장을 입력하면 다음 단어 확률을 출력
# -----------------------------------------------

# 샘플 문장 준비 → 단어장 생성 → 데이터 시퀀스화 → MiniTransformer 정의 → 학습 → 다음 단어 예측


# 라이브러리 임포트
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 샘플 문장 정의
sentences = [
    ['Hello', 'I', 'am', 'GPT'],
    ['How', 'are', 'you', 'today'],
    ['I', 'love', 'deep', 'learning'],
    ['This', 'is', 'a', 'test'],
    ['PyTorch', 'is', 'very', 'powerful'],
    ['The', 'cat', 'sat', 'here'],
    ['Please', 'show', 'me', 'attention'],
    ['Learning', 'is', 'fun', 'always'],
    ['Simple', 'data', 'for', 'demo'],
    ['Transformer', 'models', 'learn', 'patterns']
]

# 2. 단어장 생성
vocab = {word for sentence in sentences for word in sentence}  # 모든 단어 집합
word2idx = {word: idx for idx, word in enumerate(vocab)}       # 단어 → 숫자 인덱스
idx2word = {idx: word for word, idx in word2idx.items()}       # 숫자 인덱스 → 단어
vocab_size = len(vocab)                                       # 단어장 크기

# 3. 입력 시퀀스와 타겟 시퀀스 생성 (다음 단어 예측용)
input_sequences = []
target_sequences = []

for sentence in sentences:
    input_seq = [word2idx[word] for word in sentence[:-1]]    # 마지막 단어 제외 (입력)
    target_seq = [word2idx[word] for word in sentence[1:]]    # 첫 단어 제외 (정답)
    input_sequences.append(input_seq)
    target_sequences.append(target_seq)

# 텐서 변환
inputs = torch.tensor(input_sequences)
targets = torch.tensor(target_sequences)

# 4. 미니 트랜스포머 모델 정의
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_size=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)  # 단어 임베딩
        self.attention = nn.MultiheadAttention(embedding_size, num_heads=1, batch_first=True)  # Self Attention
        self.fc = nn.Linear(embedding_size, vocab_size)            # 단어장 크기로 출력 변환

    def forward(self, x):
        embedded = self.embedding(x)                               # (Batch, Seq, Embed)
        attn_output, _ = self.attention(embedded, embedded, embedded)  # Self Attention
        out = self.fc(attn_output)                                 # (Batch, Seq, Vocab)
        return out

# 5. 모델, 손실함수, 옵티마이저 설정
model = MiniTransformer(vocab_size)
criterion = nn.CrossEntropyLoss()                                 # 다중 클래스 분류용 Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 6. 학습 (500 Epoch)
for epoch in range(500):
    optimizer.zero_grad()
    output = model(inputs)                                        # (Batch, Seq, Vocab)
    loss = criterion(output.view(-1, vocab_size), targets.view(-1))  # Flatten 처리 후 Loss 계산
    loss.backward()
    optimizer.step()

    # (선택) 진행 상황 출력
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 7. 예측 함수 정의 (다음 단어 확률 분포 출력)
def predict_next_word(input_words):
    model.eval()
    input_ids = torch.tensor([[word2idx[word] for word in input_words]])  # 입력 단어를 숫자 시퀀스로 변환

    with torch.no_grad():
        output = model(input_ids)                          # 모델 예측
        last_token_output = output[0, -1]                  # 마지막 단어의 출력 (다음 단어 확률)
        probabilities = torch.softmax(last_token_output, dim=0)  # 소프트맥스 확률 분포 계산

    # 확률을 단어-확률 딕셔너리로 변환
    prob_dict = {idx2word[idx]: float(probabilities[idx]) for idx in range(vocab_size)}
    sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

    # 출력
    print(f"\n입력 문장: {' '.join(input_words)}")
    print("\n다음 단어 확률 분포:")
    for word, prob in sorted_probs.items():
        print(f"{word}: {prob:.4f}")
    print(f"\n가장 높은 확률의 다음 단어: {max(sorted_probs, key=sorted_probs.get)}")

# 8. 테스트 (문장을 입력하면 다음 단어 예측)
predict_next_word(['Hello', 'I', 'am'])
predict_next_word(['How', 'are', 'you'])
predict_next_word(['This', 'is', 'a'])
# predict_next_word(['This', 'is', 'great'])  # great은 사전에 없으므로 주의
