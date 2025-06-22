import numpy as np
import random
import time
import os
import platform

# 기본 설정
GRID_SIZE = 5
ACTIONS = ['up', 'down', 'left', 'right']
ACTION_MAP = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}

log_history = []

# 콘솔 지우기 함수
def clear_console():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        print("\033[H\033[J", end="")

# 게임판 + 로그 함께 출력
def print_grid_and_logs(snake, food):
    grid = [[' ' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for i, (x, y) in enumerate(snake):
        grid[x][y] = 'S' if i == 0 else 'o'
    fx, fy = food
    grid[fx][fy] = 'F'

    clear_console()

    # 1. 게임판 출력
    print("+" + "-" * GRID_SIZE + "+")
    for row in grid:
        print("|" + "".join(row) + "|")
    print("+" + "-" * GRID_SIZE + "+")

    # 2. 로그 출력 (아래 고정)
    print("\n--- Action Log ---")
    for line in log_history[-20:]:  # 최근 20줄만
        print(line)

# 음식 생성
def generate_food(snake):
    while True:
        f = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        if f not in snake:
            return f

# 실제 게임 시작
def run_snake_simulation():
    snake = [(2, 2)]
    food = generate_food(snake)
    done = False
    steps = 0
    total_reward = 0

    while not done and steps < 50:
        steps += 1
        action = random.randint(0, 3)
        dx, dy = ACTION_MAP[action]
        head = snake[0]
        new_head = (head[0] + dx, head[1] + dy)

        reward = -0.1
        symbol = ""
        if (
            new_head[0] < 0 or new_head[0] >= GRID_SIZE or
            new_head[1] < 0 or new_head[1] >= GRID_SIZE or
            new_head in snake
        ):
            reward = -10
            done = True
            symbol = "💥 CRASH"
        else:
            if new_head == food:
                reward = 10
                snake.insert(0, new_head)
                food = generate_food(snake)
                symbol = "🍎 ATE"
            else:
                snake.insert(0, new_head)
                snake.pop()

        total_reward += reward
        log_line = f"[{steps:2}] Action: {ACTIONS[action]:<5} | Reward: {reward:>5} {symbol}"
        log_history.append(log_line)

        print_grid_and_logs(snake, food)
        time.sleep(0.25)

    print(f"\n🎯 Simulation ended. Total reward: {total_reward:.2f}\n")

# 실행
if __name__ == "__main__":
    run_snake_simulation()
