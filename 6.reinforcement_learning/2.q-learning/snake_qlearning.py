import numpy as np
import random
import json

# 환경 설정
GRID_SIZE = 10
ACTIONS = ['up', 'down', 'left', 'right']
ACTION_MAP = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}

# Q-Table 초기화
q_table = {}

def get_state(snake_head, food):
    return f"{snake_head[0]},{snake_head[1]}_{food[0]},{food[1]}"

def choose_action(state, epsilon=0.1):
    if state not in q_table:
        q_table[state] = np.zeros(len(ACTIONS))

    if random.random() < epsilon:
        return random.randint(0, 3)
    return int(np.argmax(q_table[state]))

def update_q_table(prev_state, action, reward, next_state, alpha=0.1, gamma=0.9):
    if next_state not in q_table:
        q_table[next_state] = np.zeros(len(ACTIONS))
    predict = q_table[prev_state][action]
    target = reward + gamma * np.max(q_table[next_state])
    q_table[prev_state][action] += alpha * (target - predict)

def generate_food(snake):
    while True:
        food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        if food not in snake:
            return food

# 학습 시작
EPISODES = 5000
for episode in range(EPISODES):
    snake = [(random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))]
    food = generate_food(snake)
    total_reward = 0
    done = False
    steps = 0

    while not done and steps < 100:
        steps += 1
        state = get_state(snake[0], food)
        action = choose_action(state)
        dx, dy = ACTION_MAP[action]
        new_head = (snake[0][0] + dx, snake[0][1] + dy)

        # 기본 보상
        reward = -0.1

        # 충돌 여부
        if (
            new_head[0] < 0 or new_head[0] >= GRID_SIZE or
            new_head[1] < 0 or new_head[1] >= GRID_SIZE or
            new_head in snake
        ):
            reward = -10
            done = True
        elif new_head == food:
            reward = 10
            snake.insert(0, new_head)
            food = generate_food(snake)
        else:
            snake.insert(0, new_head)
            snake.pop()

        next_state = get_state(snake[0], food)
        update_q_table(state, action, reward, next_state)
        total_reward += reward

    if (episode + 1) % 500 == 0:
        print(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}")

# 학습된 Q-table 저장
with open("q_table.json", "w") as f:
    json.dump({k: v.tolist() for k, v in q_table.items()}, f)

print("✅ Q-learning 학습 완료! q_table.json 파일 저장됨.")
