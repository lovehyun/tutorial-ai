import geoip2.database
import requests
import ipaddress

# wget https://raw.githubusercontent.com/P3TERX/GeoLite.mmdb/download/GeoLite2-City.mmdb
reader = geoip2.database.Reader("GeoLite2-City.mmdb")

_cached_public_ip = None  # 캐시용 변수

def get_location(ip):
    global _cached_public_ip
    try:
        # 사설 IP인 경우 공인 IP로 대체
        if is_private_ip(ip):
            if _cached_public_ip is None:
                _cached_public_ip = get_public_ip()
            ip = _cached_public_ip
            if not ip:
                return None
            
        response = reader.city(ip)
        lat = response.location.latitude
        lng = response.location.longitude
        # print(f"[DEBUG] {ip} → lat={lat}, lng={lng}")
        return {"lat": lat, "lng": lng}
    except Exception as e:
        # print(f"[DEBUG] {ip} 위치 정보 없음: {e}")
        return None

# 현재 내 공인 IP 가져오기
def get_public_ip():
    try:
        return requests.get("https://api.ipify.org").text
    except:
        return None

# 사설 IP인지 확인
def is_private_ip(ip):
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return True
