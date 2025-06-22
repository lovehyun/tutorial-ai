import numpy as np
import random
import time
import os

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

# 콘솔 초기화
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

# 상태 표현
def get_state(snake, food):
    return f"{snake[0][0]},{snake[0][1]}_{food[0]},{food[1]}"

# 행동 선택
def choose_action(state, epsilon=0.1):
    if state not in q_table:
        q_table[state] = np.zeros(len(ACTIONS))
    return random.randint(0, 3) if random.random() < epsilon else int(np.argmax(q_table[state]))

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

# 학습 메인 루프
EPISODES = 1000

for episode in range(EPISODES):
    snake = [(random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))]
    food = generate_food(snake)
    total_reward = 0
    food_eaten = 0
    done = False
    step = 0

    show_this_episode = (episode + 1) % 100 == 0

    if show_this_episode:
        print(f"\n=== EPISODE {episode+1} ===")

    while not done and step < 50:
        step += 1
        state = get_state(snake, food)
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
            food_eaten += 1
            snake.insert(0, new_head)
            food = generate_food(snake)
        else:
            snake.insert(0, new_head)
            snake.pop()

        next_state = get_state(snake, food)
        update_q_table(state, action, reward, next_state)
        total_reward += reward

        if show_this_episode:
            clear_console()
            print_grid(snake, food)
            print(f"[EP {episode+1} STEP {step:>2}] Action: {ACTIONS[action]:<5} | Reward: {reward:>5} | Total: {total_reward:>5.1f} | 🍎: {food_eaten}")
            time.sleep(0.15)

    if show_this_episode:
        print(f"\n✅ Episode {episode+1} ended.")
        print(f"   ➤ Total reward: {total_reward:.2f}")
        print(f"   ➤ Steps taken : {step}")
        print(f"   ➤ Food eaten  : {food_eaten}")
        if food_eaten == 0:
            print(f"⚠️  Died without eating any food!")
        time.sleep(1.5)
