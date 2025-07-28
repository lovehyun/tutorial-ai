from scapy.arch.windows import get_windows_if_list

print("사용 가능한 인터페이스 이름 목록:")
for iface in get_windows_if_list():
    print("-", iface['name'])
