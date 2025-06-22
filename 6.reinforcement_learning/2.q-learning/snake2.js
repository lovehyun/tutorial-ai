const BLOCK_SIZE = 20;
const CANVAS_SIZE = 100;
const SNAKE_SPEED = 200;

let canvas, context;
let snake = [{ x: 0, y: 0 }];
let direction = 'right';
let food = generateFood();
let gameover = false;

// 초기화
window.onload = () => {
    canvas = document.getElementById("snakeCanvas");
    context = canvas.getContext("2d");
    setInterval(gameLoop, SNAKE_SPEED);
};

// 상태 문자열 생성 (예: "5,5_1,2")
function getCurrentState1() {
    const head = snake[0];
    return `${food.x}_${food.y},${head.x},${head.y}`;
}

// 상태 문자열 생성 (예: "5,5_1,2,1,1")
function getCurrentState2() {
    const head = snake[0];
    const bodyArr = snake.slice(1).map(seg => `${seg.x},${seg.y}`);
    const bodyStr = bodyArr.length > 0 ? ',' + bodyArr.join(',') : '';
    return `${food.x},${food.y}_${head.x},${head.y}${bodyStr}`;
}

// Q-Table에서 가장 높은 Q값을 가진 방향 선택
function getBestAction(state) {
    const q = Q_TABLE[state];
    if (!q) {
        return direction; // 모르는 상태는 기존 방향 유지
    }
    const bestIndex = q.indexOf(Math.max(...q));
    return ['up', 'down', 'left', 'right'][bestIndex];
}

// 게임 루프
function gameLoop() {
    if (!gameover) {
        const state = getCurrentState2();
        direction = getBestAction(state); // AI가 방향 결정

        const q = Q_TABLE[state];

        // ✅ 디버깅 로그 출력
        console.log("🔍 상태(state):", state);
        if (q) {
            console.log("📊 Q값들:", q.map((v, i) => `${['↑','↓','←','→'][i]}: ${v.toFixed(2)}`).join(' | '));
        } else {
            console.log("⚠️ 이 상태에 대한 Q값이 없습니다.");
        }
        console.log("👉 선택한 방향:", direction);
        console.log("🟦 머리 위치:", snake[0]);
        console.log("---------------------------");
    }

    moveSnake();
    checkCollision();
    checkFoodCollision();
    draw();
}

// 뱀 이동
function moveSnake() {
    const head = { ...snake[0] };
    switch (direction) {
        case 'up': head.y -= 1; break;
        case 'down': head.y += 1; break;
        case 'left': head.x -= 1; break;
        case 'right': head.x += 1; break;
    }
    snake.unshift(head);
}

// 충돌 확인
function checkCollision() {
    const head = snake[0];
    const max = CANVAS_SIZE / BLOCK_SIZE;

    if (head.x < 0 || head.x >= max || head.y < 0 || head.y >= max || isSnakeCollision()) {
        gameover = true;
    }
}

// 몸통 충돌 확인
function isSnakeCollision() {
    const head = snake[0];
    return snake.slice(1).some(seg => seg.x === head.x && seg.y === head.y);
}

// 음식 먹었는지 확인
function checkFoodCollision() {
    const head = snake[0];
    if (head.x === food.x && head.y === food.y) {
        food = generateFood();
    } else {
        snake.pop(); // 안 먹었으면 꼬리 줄이기
    }
}

// 음식 생성
function generateFood() {
    let position;
    do {
        position = {
            x: Math.floor(Math.random() * (CANVAS_SIZE / BLOCK_SIZE)),
            y: Math.floor(Math.random() * (CANVAS_SIZE / BLOCK_SIZE)),
        };
    } while (snake.some(seg => seg.x === position.x && seg.y === position.y));
    return position;
}

// 그리기
function draw() {
    context.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    if (gameover) {
        context.fillStyle = "#F00";
        context.font = "30px Arial";
        context.fillText("Game Over", 80, CANVAS_SIZE / 2);
        return;
    }

    drawSnake();
    drawFood();
}

function drawSnake() {
    snake.forEach((seg, i) => {
        context.fillStyle = i === 0 ? "#0077FF" : "#0055AA";
        context.fillRect(seg.x * BLOCK_SIZE, seg.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    });
}

function drawFood() {
    context.fillStyle = "#F00";
    context.fillRect(food.x * BLOCK_SIZE, food.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
}
