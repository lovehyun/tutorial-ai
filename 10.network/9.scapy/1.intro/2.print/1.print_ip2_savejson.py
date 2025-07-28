from scapy.all import sniff
import json

results = []

def extract_features(pkt):
    if pkt.haslayer("IP"):
        data = {
            "src": pkt["IP"].src,
            "dst": pkt["IP"].dst,
            "proto": pkt["IP"].proto,
            "len": len(pkt)
        }
        results.append(data)

sniff(prn=extract_features, count=10)

# 결과 저장
with open("packets.json", "w") as f:
    # json.dump(results, f, indent=2)

    # 한줄에 하나씩 (JSONL 형식)
    for pkt_data in results:
        f.write(json.dumps(pkt_data) + "\n")
