const socket = io();
const log = document.getElementById("log");

socket.on("realtime_result", (data) => {
    const li = document.createElement("li");
    li.textContent = `예측: ${data.label} (확률: ${data.score})`;
    if (data.label === "이상") li.style.color = "red";
    log.prepend(li);
});
