# 포트폴리오 최적화 시스템

## 🎯 프로젝트 개요
효율적 경계선 이론과 현대 포트폴리오 이론을 기반으로 한 과학적 투자 전략 도구

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 패키지 설치
pip install -r requirements.txt
```

### 2. 샘플 데이터 생성
```bash
python create_sample_data.py
```

### 3. 서버 실행
```bash
python app.py
```

### 4. 웹 브라우저에서 접속
http://localhost:5000

## 📊 주요 기능
- ✅ 효율적 경계선 계산
- ✅ 최대 샤프비율 포트폴리오
- ✅ 최소분산 포트폴리오  
- ✅ 몬테카를로 시뮬레이션
- ✅ 2종목 다각화 분석
- ✅ 상관관계 분석
- ✅ 투자 시뮬레이션

## 📁 파일 구조
```
portfolio-optimizer/
├── app.py                 # Flask 백엔드 서버
├── create_sample_data.py   # 샘플 데이터 생성
├── requirements.txt        # Python 패키지
├── data/                   # 데이터 파일들
│   ├── korean_stocks_sample.csv
│   ├── global_etf_sample.csv
│   └── sector_rotation_sample.csv
└── README.md
```

## 🔧 사용법
1. 데이터 파일을 data/ 폴더에 배치
2. 웹 인터페이스에서 파일 선택
3. 포트폴리오 최적화 실행
4. 결과 분석 및 투자 전략 수립

## 📈 분석 결과
- 효율적 경계선 차트
- 최적 포트폴리오 배분
- 위험-수익률 분석
- 다각화 효과 측정

## ⚠️ 주의사항
- 이 도구는 교육 및 연구 목적입니다
- 실제 투자 결정 시 전문가 상담 권장
- 과거 데이터 기반 분석의 한계 인지 필요
