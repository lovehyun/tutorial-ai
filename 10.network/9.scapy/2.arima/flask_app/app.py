from flask import Flask, render_template, jsonify
from packet_collector import PacketMonitor

app = Flask(__name__)
monitor = PacketMonitor()
monitor.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    data = {}
    for label, rule in [('second', '1s'), ('minute', '1min'), ('hour', '1h')]:
        df = monitor.get_resampled(rule)
        if df.empty:
            continue
        recent, forecast, lower, upper = monitor.get_forecast(df['count'])
        data[label] = {
            'timestamps': df.index[-10:].strftime('%H:%M:%S').tolist(),
            'recent': recent,
            'forecast': forecast,
            'lower': lower,
            'upper': upper
        }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
