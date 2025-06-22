import numpy as np
import random

# 환경 설정
n_states = 5            # 상태 공간: 위치 0~4
actions = [0, 1]        # 0: 왼쪽, 1: 오른쪽
q_table = np.zeros((n_states, len(actions)))

# 하이퍼파라미터
alpha = 0.1             # 학습률
gamma = 0.9             # 할인율
epsilon = 0.3           # 탐험 비율
episodes = 100

# 보상 정의
def get_reward(state):
    return 1 if state == 4 else 0

# 행동 선택 (탐험 vs 활용)
def choose_action(state):
    if random.random() < epsilon:
        return random.choice(actions)  # 탐험
    return np.argmax(q_table[state])  # 활용

# 상태 전이
def take_action(state, action):
    if action == 0:
        return max(0, state - 1)
    else:
        return min(n_states - 1, state + 1)

# 학습 시작
for episode in range(episodes):
    state = 0  # 시작 위치
    done = False

    while not done:
        action = choose_action(state)
        next_state = take_action(state, action)
        reward = get_reward(next_state)

        # Q-테이블 업데이트
        q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

        state = next_state
        if state == 4:  # 목표 도달
            done = True

print("학습된 Q-테이블:")
print(np.round(q_table, 2))
