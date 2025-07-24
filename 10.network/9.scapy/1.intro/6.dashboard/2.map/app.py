# pip install flask scapy geoip2

from flask import Flask, render_template, jsonify
from packet_collector import collector
from geoip_utils import get_location

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/packets')
def packets():
    data = collector.get_packets()
    print(f"[DEBUG] 현재 수집된 패킷 수: {len(data)}")

    lines = []
    for pkt in data:
        src = get_location(pkt['sip'])
        dst = get_location(pkt['dip'])
        
        if src and dst:
            lines.append({
                'src': src,
                'dst': dst,
                'proto': pkt['proto']
            })
            
    print(f"[DEBUG] 위치 정보가 포함된 패킷 수: {len(lines)}")

    return jsonify(lines)

if __name__ == '__main__':
    collector.start()
    app.run(debug=True)
