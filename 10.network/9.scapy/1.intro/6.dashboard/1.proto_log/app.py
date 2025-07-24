from flask import Flask, render_template, jsonify
from packet_collector import PacketCollector

app = Flask(__name__)
collector = PacketCollector()
collector.start()

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/data')
def data():
    stats = collector.get_stats()
    return jsonify(stats)

if __name__ == '__main__':
    app.run(debug=True)
