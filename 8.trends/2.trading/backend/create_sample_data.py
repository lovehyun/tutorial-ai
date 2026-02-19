# create_sample_data.py - 샘플 데이터 생성 스크립트
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_korean_stocks_data():
    """삼성전자, LG전자, SK텔레콤, KB금융, 신한지주 샘플 데이터 생성"""
    
    print("🏢 한국 주요 종목 샘플 데이터 생성 중...")
    
    # 종목 리스트
    stocks = ['삼성전자', 'LG전자', 'SK텔레콤', 'KB금융', '신한지주']
    
    # 기간 설정 (최근 2년)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    # 거래일만 생성 (주말 제외)
    dates = pd.bdate_range(start=start_date, end=end_date)
    
    # 각 종목별 실제 특성 반영
    stock_params = {
        '삼성전자': {
            'base_price': 70000, 
            'volatility': 0.28, 
            'trend': 0.08,
            'sector': 'IT'
        },
        'LG전자': {
            'base_price': 120000, 
            'volatility': 0.32, 
            'trend': 0.06,
            'sector': 'IT'
        },
        'SK텔레콤': {
            'base_price': 50000, 
            'volatility': 0.22, 
            'trend': 0.04,
            'sector': '통신'
        },
        'KB금융': {
            'base_price': 55000, 
            'volatility': 0.35, 
            'trend': 0.10,
            'sector': '금융'
        },
        '신한지주': {
            'base_price': 45000, 
            'volatility': 0.33, 
            'trend': 0.09,
            'sector': '금융'
        }
    }
    
    # 실제 한국 주식시장 상관관계 반영
    correlation_matrix = np.array([
        [1.00, 0.65, 0.45, 0.55, 0.52],  # 삼성전자
        [0.65, 1.00, 0.42, 0.48, 0.45],  # LG전자  
        [0.45, 0.42, 1.00, 0.38, 0.35],  # SK텔레콤
        [0.55, 0.48, 0.38, 1.00, 0.75],  # KB금융
        [0.52, 0.45, 0.35, 0.75, 1.00]   # 신한지주
    ])
    
    # 시드 설정으로 재현 가능한 데이터
    np.random.seed(42)
    n_days = len(dates)
    
    # 일간 기대수익률 (연간을 252 거래일로 나눔)
    daily_returns_mean = [stock_params[stock]['trend']/252 for stock in stocks]
    
    # 일간 변동성 행렬
    daily_vol_matrix = correlation_matrix * np.outer(
        [stock_params[stock]['volatility']/np.sqrt(252) for stock in stocks],
        [stock_params[stock]['volatility']/np.sqrt(252) for stock in stocks]
    )
    
    # 상관관계를 고려한 다변량 정규분포에서 수익률 생성
    returns = np.random.multivariate_normal(
        mean=daily_returns_mean,
        cov=daily_vol_matrix,
        size=n_days
    )
    
    # 누적 수익률로 가격 계산
    prices = {}
    for i, stock in enumerate(stocks):
        base_price = stock_params[stock]['base_price']
        # 시장 충격과 계절성 효과 추가
        market_shocks = add_market_events(returns[:, i], dates)
        cumulative_returns = np.cumprod(1 + market_shocks)
        prices[stock] = base_price * cumulative_returns
    
    # DataFrame 생성
    df = pd.DataFrame(prices, index=dates)
    
    # 데이터 디렉토리 생성
    os.makedirs('data', exist_ok=True)
    
    # CSV 파일로 저장
    filename = 'data/korean_stocks_sample.csv'
    df.to_csv(filename)
    
    print(f"✅ 한국 주식 샘플 데이터 생성 완료!")
    print(f"   📁 파일: {filename}")
    print(f"   📅 기간: {dates[0].date()} ~ {dates[-1].date()}")
    print(f"   🏢 종목: {', '.join(stocks)}")
    print(f"   📊 크기: {df.shape[0]}일 × {df.shape[1]}종목")
    
    return df, stock_params

def add_market_events(returns, dates):
    """실제 시장 이벤트 효과 추가"""
    enhanced_returns = returns.copy()
    
    # 주요 시장 이벤트 시뮬레이션
    for i, date in enumerate(dates):
        # 월말 효과 (포트폴리오 리밸런싱)
        if date.day >= 25:
            enhanced_returns[i] += np.random.normal(0, 0.005)
        
        # 분기말 효과
        if date.month % 3 == 0 and date.day >= 25:
            enhanced_returns[i] += np.random.normal(0, 0.01)
        
        # 코로나19 효과 (2020년 3월)
        if date.year == 2020 and date.month == 3:
            enhanced_returns[i] += np.random.normal(-0.02, 0.03)
        
        # 연말 랠리 효과
        if date.month == 12:
            enhanced_returns[i] += np.random.normal(0.001, 0.002)
    
    return enhanced_returns

def create_global_etf_data():
    """글로벌 ETF 샘플 데이터 생성"""
    
    print("\n🌍 글로벌 ETF 샘플 데이터 생성 중...")
    
    etfs = ['S&P500_ETF', 'NASDAQ_ETF', '유럽_ETF', '신흥국_ETF', '채권_ETF']
    
    # ETF별 특성
    etf_params = {
        'S&P500_ETF': {'base_price': 100, 'volatility': 0.16, 'trend': 0.10},
        'NASDAQ_ETF': {'base_price': 120, 'volatility': 0.22, 'trend': 0.12},
        '유럽_ETF': {'base_price': 85, 'volatility': 0.18, 'trend': 0.06},
        '신흥국_ETF': {'base_price': 95, 'volatility': 0.25, 'trend': 0.08},
        '채권_ETF': {'base_price': 90, 'volatility': 0.05, 'trend': 0.03}
    }
    
    # 글로벌 자산간 상관관계
    etf_correlation = np.array([
        [1.00, 0.85, 0.70, 0.65, -0.15],  # S&P500
        [0.85, 1.00, 0.68, 0.70, -0.20],  # NASDAQ
        [0.70, 0.68, 1.00, 0.75, -0.10],  # 유럽
        [0.65, 0.70, 0.75, 1.00, -0.05],  # 신흥국
        [-0.15, -0.20, -0.10, -0.05, 1.00]  # 채권
    ])
    
    # 같은 기간으로 데이터 생성
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    dates = pd.bdate_range(start=start_date, end=end_date)
    
    np.random.seed(123)
    n_days = len(dates)
    
    daily_returns_mean = [etf_params[etf]['trend']/252 for etf in etfs]
    daily_vol_matrix = etf_correlation * np.outer(
        [etf_params[etf]['volatility']/np.sqrt(252) for etf in etfs],
        [etf_params[etf]['volatility']/np.sqrt(252) for etf in etfs]
    )
    
    returns = np.random.multivariate_normal(
        mean=daily_returns_mean,
        cov=daily_vol_matrix,
        size=n_days
    )
    
    prices = {}
    for i, etf in enumerate(etfs):
        base_price = etf_params[etf]['base_price']
        cumulative_returns = np.cumprod(1 + returns[:, i])
        prices[etf] = base_price * cumulative_returns
    
    df = pd.DataFrame(prices, index=dates)
    filename = 'data/global_etf_sample.csv'
    df.to_csv(filename)
    
    print(f"✅ 글로벌 ETF 샘플 데이터 생성 완료!")
    print(f"   📁 파일: {filename}")
    print(f"   🌍 ETF: {', '.join(etfs)}")
    
    return df

def create_sector_rotation_data():
    """섹터 로테이션 분석용 데이터 생성"""
    
    print("\n🔄 섹터 로테이션 샘플 데이터 생성 중...")
    
    sectors = ['기술주', '금융주', '헬스케어', '에너지', '소비재', '산업재', '유틸리티', '부동산']
    
    sector_params = {
        '기술주': {'base_price': 150, 'volatility': 0.30, 'trend': 0.15},
        '금융주': {'base_price': 80, 'volatility': 0.28, 'trend': 0.08},
        '헬스케어': {'base_price': 120, 'volatility': 0.20, 'trend': 0.10},
        '에너지': {'base_price': 60, 'volatility': 0.35, 'trend': 0.05},
        '소비재': {'base_price': 100, 'volatility': 0.18, 'trend': 0.07},
        '산업재': {'base_price': 90, 'volatility': 0.25, 'trend': 0.09},
        '유틸리티': {'base_price': 70, 'volatility': 0.15, 'trend': 0.04},
        '부동산': {'base_price': 85, 'volatility': 0.22, 'trend': 0.06}
    }
    
    # 섹터간 상관관계 (경기순환을 반영)
    np.random.seed(789)
    sector_correlation = np.random.uniform(0.2, 0.7, (len(sectors), len(sectors)))
    np.fill_diagonal(sector_correlation, 1.0)
    # 대칭 행렬로 만들기
    sector_correlation = (sector_correlation + sector_correlation.T) / 2
    np.fill_diagonal(sector_correlation, 1.0)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1095)  # 3년 데이터
    dates = pd.bdate_range(start=start_date, end=end_date)
    
    n_days = len(dates)
    daily_returns_mean = [sector_params[sector]['trend']/252 for sector in sectors]
    daily_vol_matrix = sector_correlation * np.outer(
        [sector_params[sector]['volatility']/np.sqrt(252) for sector in sectors],
        [sector_params[sector]['volatility']/np.sqrt(252) for sector in sectors]
    )
    
    returns = np.random.multivariate_normal(
        mean=daily_returns_mean,
        cov=daily_vol_matrix,
        size=n_days
    )
    
    # 섹터 로테이션 효과 추가
    for i, date in enumerate(dates):
        # 경기 사이클에 따른 섹터 성과 조정
        cycle_phase = (date.year * 12 + date.month) % 48  # 4년 주기
        
        if cycle_phase < 12:  # 회복기 - 기술주, 소비재 강세
            returns[i][0] += 0.0005  # 기술주
            returns[i][4] += 0.0003  # 소비재
        elif cycle_phase < 24:  # 확장기 - 산업재, 에너지 강세
            returns[i][5] += 0.0004  # 산업재
            returns[i][3] += 0.0003  # 에너지
        elif cycle_phase < 36:  # 둔화기 - 헬스케어, 유틸리티 강세
            returns[i][2] += 0.0003  # 헬스케어
            returns[i][6] += 0.0002  # 유틸리티
        else:  # 수축기 - 금융주 상대적 강세
            returns[i][1] += 0.0002  # 금융주
    
    prices = {}
    for i, sector in enumerate(sectors):
        base_price = sector_params[sector]['base_price']
        cumulative_returns = np.cumprod(1 + returns[:, i])
        prices[sector] = base_price * cumulative_returns
    
    df = pd.DataFrame(prices, index=dates)
    filename = 'data/sector_rotation_sample.csv'
    df.to_csv(filename)
    
    print(f"✅ 섹터 로테이션 샘플 데이터 생성 완료!")
    print(f"   📁 파일: {filename}")
    print(f"   🔄 섹터: {', '.join(sectors)}")
    
    return df

def generate_summary_report(korean_df, korean_params):
    """생성된 데이터 요약 리포트"""
    
    print("\n" + "="*60)
    print("📋 데이터 생성 요약 리포트")
    print("="*60)
    
    # 기본 통계
    print(f"\n📊 기본 통계 (한국 주식):")
    returns = korean_df.pct_change().dropna()
    annual_returns = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    sharpe_ratios = annual_returns / annual_volatility
    
    for stock in korean_df.columns:
        params = korean_params[stock]
        print(f"  {stock:8s}: 수익률 {annual_returns[stock]:6.2%}, "
              f"변동성 {annual_volatility[stock]:6.2%}, "
              f"샤프 {sharpe_ratios[stock]:5.2f}, "
              f"섹터 {params['sector']}")
    
    # 상관관계 분석
    print(f"\n🔗 상관관계 분석:")
    corr_matrix = returns.corr()
    print("상관계수 매트릭스:")
    print(corr_matrix.round(3))
    
    # 다각화 효과 분석
    print(f"\n🎯 다각화 효과:")
    equal_weight_return = annual_returns.mean()
    equal_weight_vol = np.sqrt(np.dot(np.ones(len(returns.columns))/len(returns.columns), 
                                     np.dot(returns.cov() * 252, np.ones(len(returns.columns))/len(returns.columns))))
    individual_avg_vol = annual_volatility.mean()
    
    print(f"  균등가중 포트폴리오 수익률: {equal_weight_return:.2%}")
    print(f"  균등가중 포트폴리오 변동성: {equal_weight_vol:.2%}")
    print(f"  개별 종목 평균 변동성: {individual_avg_vol:.2%}")
    print(f"  다각화 효과: {(individual_avg_vol - equal_weight_vol)/individual_avg_vol:.1%} 위험 감소")
    
    print(f"\n✅ 모든 샘플 데이터 생성 완료!")
    print(f"   📁 data/ 폴더에 CSV 파일들이 저장되었습니다.")
    print(f"   🚀 이제 Flask 백엔드를 실행하고 웹 인터페이스에서 분석을 시작하세요!")

def main():
    """메인 실행 함수"""
    print("🚀 포트폴리오 최적화용 샘플 데이터 생성기")
    print("="*50)
    
    # 1. 한국 주식 데이터 (메인)
    korean_df, korean_params = create_korean_stocks_data()
    
    # 2. 글로벌 ETF 데이터 
    global_df = create_global_etf_data()
    
    # 3. 섹터 로테이션 데이터
    sector_df = create_sector_rotation_data()
    
    # 4. 요약 리포트
    generate_summary_report(korean_df, korean_params)
    
    # 5. 추가 유틸리티 파일들 생성
    create_utility_files()

def create_utility_files():
    """추가 유틸리티 파일들 생성"""
    
    # requirements.txt 생성
    requirements = """Flask==2.3.3
Flask-CORS==4.0.0
pandas==2.1.1
numpy==1.24.3
scipy==1.11.3
openpyxl==3.1.2
python-dateutil==2.8.2
matplotlib==3.7.2
seaborn==0.12.2
yfinance==0.2.18
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    print(f"\n📦 requirements.txt 생성 완료")
    
    # .gitignore 생성
    gitignore = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Data files (optional - uncomment if you don't want to track data)
# data/*.csv
# data/*.xlsx

# Logs
*.log

# React
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
build/
.env.local
.env.development.local
.env.test.local
.env.production.local
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore)
    print(f"📝 .gitignore 생성 완료")
    
    # README.md 생성
    readme = """# 포트폴리오 최적화 시스템

## 🎯 프로젝트 개요
효율적 경계선 이론과 현대 포트폴리오 이론을 기반으로 한 과학적 투자 전략 도구

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate     # Windows

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
"""
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme)
    print(f"📚 README.md 생성 완료")

if __name__ == "__main__":
    main()
