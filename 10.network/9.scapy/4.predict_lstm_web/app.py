from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

@app.route("/")
def index():
    return render_template("index.html")

# 패킷 모니터로부터 결과 수신 후 브로드캐스트
@socketio.on("result_from_packet")
def handle_result(data):
    socketio.emit("realtime_result", data)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)
