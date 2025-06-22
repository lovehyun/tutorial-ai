import json

# 입력 파일 (Python에서 저장한 Q-table)
input_file = "q_table.json"

# 출력 파일 (JS에서 불러올 Q-table)
output_file = "q_table.js"

with open(input_file, "r") as f:
    q_table = json.load(f)

# JS 형식으로 파일 작성
with open(output_file, "w", encoding="utf-8") as f:
    f.write("const Q_TABLE = ")
    json.dump(q_table, f, indent=2)
    f.write(";")  # 끝에 세미콜론 추가
    print(f"✅ '{output_file}' 파일이 성공적으로 생성되었습니다!")
