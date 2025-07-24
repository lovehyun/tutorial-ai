# test_geoip.py
from geoip_utils import get_location

# 실제 외부 IP 주소로 테스트 (예: 구글 DNS)
test_ips = [
    "8.8.8.8",        # Google DNS
    "1.1.1.1",        # Cloudflare DNS
    "114.111.34.34",  # 한국 통신사 (KT 등)
    "192.168.0.1",    # 사설 IP (결과는 None)
]

for ip in test_ips:
    loc = get_location(ip)
    if loc and loc.get("lat") is not None and loc.get("lng") is not None:
        print(f"[{ip}] 위치: 위도={loc['lat']:.4f}, 경도={loc['lng']:.4f}")
    else:
        print(f"[{ip}] 위치 정보 없음 (사설 IP 또는 DB에 없음)")
