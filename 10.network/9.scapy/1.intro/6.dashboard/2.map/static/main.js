const map = L.map('map').setView([20, 0], 2);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);

// 매번 그려지는 원들을 저장 (필요시)
let circles = [];

/**
 * 점이 퍼지듯 확산되는 원 그리기
 */
function drawPulse(lat, lng, color = 'red') {
    if (lat == null || lng == null) return;

    const circle = L.circle([lat, lng], {
        color: color,
        fillColor: color,
        fillOpacity: 0.6,
        radius: 20000,
    }).addTo(map);

    circles.push(circle); // 원 관리용 (선택)

    let radius = 20000;
    let opacity = 0.6;

    const interval = setInterval(() => {
        radius += 10000;
        opacity -= 0.05;

        if (opacity <= 0) {
            map.removeLayer(circle);
            clearInterval(interval);
        } else {
            circle.setRadius(radius);
            circle.setStyle({ fillOpacity: opacity, opacity: opacity });
        }
    }, 100);
}

function drawDashedLine(lat1, lng1, lat2, lng2, color = 'gray') {
    if ([lat1, lng1, lat2, lng2].some(v => v == null)) return;

    const line = L.polyline([[lat1, lng1], [lat2, lng2]], {
        color: color,
        dashArray: '5, 10', // 점선 스타일: 선5, 공간10
        weight: 2,
        opacity: 0.7,
    }).addTo(map);

    // 5초 후 자동 제거
    setTimeout(() => {
        map.removeLayer(line);
    }, 5000);
}

function drawArcLine(lat1, lng1, lat2, lng2, color = 'purple') {
    if ([lat1, lng1, lat2, lng2].some(v => v == null)) return;

    const arcLine = L.Polyline.Arc(
        [lat1, lng1],
        [lat2, lng2],
        { color: color, weight: 2, dashArray: '4,8', opacity: 0.7 }
    ).addTo(map);

    // 자동 제거 (예: 5초 후)
    setTimeout(() => map.removeLayer(arcLine), 5000);
}

/**
 * 서버에서 패킷 데이터 가져와 시각화
 */
function fetchPackets() {
    fetch('/packets')
        .then((res) => res.json())
        .then((data) => {
            data.forEach((pkt) => {
                const src = pkt.src;
                const dst = pkt.dst;

                if (
                    src && dst &&
                    src.lat != null && src.lng != null &&
                    dst.lat != null && dst.lng != null
                ) {
                    drawPulse(src.lat, src.lng, 'blue');   // 출발지
                    drawPulse(dst.lat, dst.lng, 'orange'); // 목적지
                    drawDashedLine(src.lat, src.lng, dst.lat, dst.lng); // 점선
                    // drawArcLine(src.lat, src.lng, dst.lat, dst.lng);  // 포물선 연결
                }
            });
        })
        .catch((err) => {
            console.error('패킷 가져오기 실패:', err);
        });
}

// 1초마다 갱신
setInterval(fetchPackets, 1000);
