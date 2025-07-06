import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 샘플 문장 정의
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

# 단어장 생성 + UNK 추가
vocab = {word for sentence in sentences for word in sentence}
vocab.add('UNK')
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(vocab)

# 데이터 준비
input_sequences = []
target_sequences = []

for sentence in sentences:
    input_seq = [word2idx[word] for word in sentence[:-1]]
    target_seq = [word2idx[word] for word in sentence[1:]]
    input_sequences.append(input_seq)
    target_sequences.append(target_seq)

inputs = torch.tensor(input_sequences)
targets = torch.tensor(target_sequences)

# 미니 트랜스포머 정의
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_size=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.attention = nn.MultiheadAttention(embedding_size, num_heads=1, batch_first=True)
        self.fc = nn.Linear(embedding_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        attn_output, _ = self.attention(embedded, embedded, embedded)
        out = self.fc(attn_output)
        return out

# 모델 정의
model = MiniTransformer(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 학습
for epoch in range(500):
    optimizer.zero_grad()
    output = model(inputs)
    loss = criterion(output.view(-1, vocab_size), targets.view(-1))
    loss.backward()
    optimizer.step()

# 예측 함수
def predict_next_word(input_words):
    model.eval()
    input_ids = []
    for word in input_words:
        if word in word2idx:
            input_ids.append(word2idx[word])
        else:
            input_ids.append(word2idx['UNK'])  # 없는 단어는 UNK 처리

    input_ids = torch.tensor([input_ids])

    with torch.no_grad():
        output = model(input_ids)
        last_token_output = output[0, -1]
        probabilities = torch.softmax(last_token_output, dim=0)

    prob_dict = {idx2word[idx]: float(probabilities[idx]) for idx in range(vocab_size)}
    sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

    print(f"\n입력 문장: {' '.join(input_words)}")
    print("\n다음 단어 확률 분포:")
    for word, prob in sorted_probs.items():
        print(f"{word}: {prob:.4f}")
    print(f"\n가장 높은 확률의 다음 단어: {max(sorted_probs, key=sorted_probs.get)}")

# 테스트: 없는 단어 포함
predict_next_word(['This', 'is', 'great'])  # 'great'은 없는 단어
predict_next_word(['PyTorch', 'is', 'awesome'])  # 'awesome'도 없는 단어
