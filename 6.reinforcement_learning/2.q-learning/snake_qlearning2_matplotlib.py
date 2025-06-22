import numpy as np
import random
import json
import matplotlib.pyplot as plt
from collections import defaultdict

# 환경 설정
GRID_SIZE = 10
ACTIONS = ['up', 'down', 'left', 'right']
ACTION_MAP = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}

q_table = {}
visit_count = defaultdict(int)

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
    q_table[prev_state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[prev_state][action])

def generate_food(snake):
    while True:
        food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        if food not in snake:
            return food

# 학습
EPISODES = 3000
reward_history = []

for episode in range(EPISODES):
    snake = [(random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))]
    food = generate_food(snake)
    total_reward = 0
    steps = 0
    done = False

    while not done and steps < 100:
        steps += 1
        state = get_state(snake[0], food)
        visit_count[state] += 1
        action = choose_action(state)
        dx, dy = ACTION_MAP[action]
        new_head = (snake[0][0] + dx, snake[0][1] + dy)

        reward = -0.1
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

    reward_history.append(total_reward)

    if (episode + 1) % 500 == 0:
        print(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}")

# 저장
with open("q_table.json", "w") as f:
    json.dump({k: v.tolist() for k, v in q_table.items()}, f)

print("✅ Q-table 저장 완료!")

# ------------------------
# ✅ 시각화
# ------------------------

# 1. 에피소드별 보상 그래프
plt.figure(figsize=(10, 4))
plt.plot(reward_history)
plt.title("Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. 가장 많이 방문한 상태 Top 10
top_states = sorted(visit_count.items(), key=lambda x: x[1], reverse=True)[:10]
states, counts = zip(*top_states)

plt.figure(figsize=(10, 4))
plt.barh(states, counts, color='skyblue')
plt.title("Top 10 Most Visited States")
plt.xlabel("Visit Count")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
