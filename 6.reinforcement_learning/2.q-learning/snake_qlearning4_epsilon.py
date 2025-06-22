import numpy as np
import random
import time
import os
import json
import matplotlib.pyplot as plt

# 환경 설정
GRID_SIZE = 5
ACTIONS = ['up', 'down', 'left', 'right']
ACTION_MAP = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}

q_table = {}
reward_history = []
food_history = []

# 콘솔 초기화
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

# 상태 표현 (머리와 음식만 가지고 표현)
def get_state1(snake, food):
    return f"{food[0]},{food[1]}_{snake[0][0]},{snake[0][1]}"

# 상태 표현 (머리와 몸통, 음식을 가지고 표현)
def get_state2(snake, food):
    head = snake[0]
    body = set(snake[1:])  # 머리 제외한 몸
    if body:
        body_key = "," + ",".join([f"{x},{y}" for (x, y) in body])
    else:
        body_key = ""
    return f"{food[0]},{food[1]}_{head[0]},{head[1]}{body_key}"

# 행동 선택
def choose_action(state, epsilon=0.1):
    if state not in q_table:
        q_table[state] = np.zeros(len(ACTIONS))
    if random.random() < epsilon:
        return random.randint(0, 3)
    else:
        return int(np.argmax(q_table[state]))

# Q 테이블 갱신
def update_q_table(s, a, r, s_, alpha=0.1, gamma=0.9):
    if s_ not in q_table:
        q_table[s_] = np.zeros(len(ACTIONS))
    q_table[s][a] += alpha * (r + gamma * np.max(q_table[s_]) - q_table[s][a])

# 음식 생성
def generate_food(snake):
    while True:
        f = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        if f not in snake:
            return f

# 격자 출력
def print_grid(snake, food):
    grid = [[' ' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for i, (x, y) in enumerate(snake):
        grid[x][y] = 'S' if i == 0 else 'o'
    fx, fy = food
    grid[fx][fy] = 'F'
    print("+" + "-" * GRID_SIZE + "+")
    for row in grid:
        print("|" + "".join(row) + "|")
    print("+" + "-" * GRID_SIZE + "+")

# Q 테이블 저장
def save_q_table(filename="q_table.json"):
    serializable_q = {state: q.tolist() for state, q in q_table.items()}
    with open(filename, "w") as f:
        json.dump(serializable_q, f)

# Q 테이블 불러오기
def load_q_table(filename="q_table.json"):
    with open(filename, "r") as f:
        loaded = json.load(f)
    return {state: np.array(q_values) for state, q_values in loaded.items()}

# JS 파일로 변환
def convert_q_table_to_js(json_file="q_table.json", js_file="q_table.js"):
    with open(json_file, "r") as f:
        q_data = json.load(f)
    with open(js_file, "w") as f:
        f.write("const Q_TABLE = {\n")
        for state, q_values in q_data.items():
            q_str = ", ".join(f"{v:.4f}" for v in q_values)
            f.write(f'  "{state}": [{q_str}],\n')
        f.write("};\n")

# 학습 메인 루프
EPISODES = 1_000_000

for episode in range(EPISODES):
    # snake = [(random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))]
    snake = [(0, 0)]
    food = generate_food(snake)
    total_reward = 0
    food_eaten = 0
    done = False
    step = 0

    epsilon = max(0.05, 1.0 - episode / (EPISODES * 0.8))
    show_this_episode = (episode + 1) % 10_000 == 0

    if show_this_episode:
        print(f"\n=== EPISODE {episode+1} ===")

    while not done and step < 50:
        step += 1
        state = get_state2(snake, food)
        action = choose_action(state, epsilon)
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
            food_eaten += 1
            snake.insert(0, new_head)
            food = generate_food(snake)
        else:
            snake.insert(0, new_head)
            snake.pop()

        next_state = get_state2(snake, food)
        update_q_table(state, action, reward, next_state)
        total_reward += reward

        if show_this_episode:
            clear_console()
            print_grid(snake, food)
            print(f"[EP {episode+1} STEP {step:>2}] Action: {ACTIONS[action]:<5} | Reward: {reward:>5} | Total: {total_reward:>5.1f} | 🍎: {food_eaten}")
            time.sleep(0.1)

    reward_history.append(total_reward)
    food_history.append(food_eaten)

    if show_this_episode:
        print(f"\n✅ Episode {episode+1} ended.")
        print(f"   ➤ Total reward: {total_reward:.2f}")
        print(f"   ➤ Steps taken : {step}")
        print(f"   ➤ Food eaten  : {food_eaten}")
        if food_eaten == 0:
            print(f"⚠️  Died without eating any food!")
        time.sleep(1.5)

# Q 테이블 저장 및 JS용 변환
save_q_table("q_table.json")
# convert_q_table_to_js("q_table.json", "q_table.js")

# 학습 결과 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(reward_history, label="Total Reward")
plt.title("Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(food_history, label="Food Eaten", color='orange')
plt.title("Food Eaten per Episode")
plt.xlabel("Episode")
plt.ylabel("Count")
plt.grid()

plt.tight_layout()
plt.show()
