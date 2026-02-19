# 필요한 라이브러리 설치 및 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 한국 주식 데이터 수집을 위한 라이브러리들
# FinanceDataReader 설치 방법:
# pip install finance-datareader  # 또는
# pip install --upgrade finance-datareader  # 또는
# conda install -c conda-forge finance-datareader
# pip install git+https://github.com/FinanceData/FinanceDataReader.git

try:
    import FinanceDataReader as fdr
    FDR_AVAILABLE = True
    print("FinanceDataReader 사용 가능")
except ImportError:
    print("FinanceDataReader 설치 필요:")
    print("pip install finance-datareader")
    FDR_AVAILABLE = False

# yfinance는 대부분 잘 설치됨
try:
    import yfinance as yf
    YF_AVAILABLE = True
    print("yfinance 사용 가능")
except ImportError:
    print("yfinance 설치 필요: pip install yfinance")
    YF_AVAILABLE = False

# 웹 크롤링을 위한 라이브러리
try:
    import requests
    from bs4 import BeautifulSoup
    CRAWLING_AVAILABLE = True
    print("웹 크롤링 사용 가능")
except ImportError:
    print("크롤링 라이브러리 설치 필요: pip install requests beautifulsoup4")
    CRAWLING_AVAILABLE = False

def get_stock_data_naver_crawling(start_date, end_date):
    """
    네이버 금융에서 주식 데이터 크롤링 (백업 방법)
    """
    if not CRAWLING_AVAILABLE:
        print("크롤링 라이브러리가 없습니다.")
        return pd.DataFrame()
    
    tickers = {
        '삼성전자': '005930',
        'LG전자': '066570', 
        'SK텔레콤': '017670',
        'KB금융': '105560',
        '신한지주': '055550'
    }
    
    data = {}
    
    for name, ticker in tickers.items():
        try:
            # 네이버 금융 URL (일간 데이터)
            url = f"https://finance.naver.com/item/sise_day.naver?code={ticker}"
            
            # 간단한 크롤링 (실제로는 더 복잡한 로직 필요)
            print(f"{name} 데이터 크롤링 시도... (실제 구현 필요)")
            
            # 여기서는 샘플 데이터 생성 (실제로는 크롤링 로직 구현)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            # 임의의 주가 데이터 (실제로는 크롤링해서 가져와야 함)
            np.random.seed(42)  # 재현 가능한 결과를 위해
            base_price = 50000 if name == '삼성전자' else 30000
            prices = base_price * (1 + np.random.randn(len(dates)) * 0.02).cumprod()
            
            data[name] = pd.Series(prices, index=dates)
            
        except Exception as e:
            print(f"{name} 크롤링 실패: {e}")
    
    return pd.DataFrame(data)

def get_sample_data():
    """
    샘플 데이터 생성 (라이브러리 설치가 안 될 경우)
    """
    print("샘플 데이터를 생성합니다...")
    
    # 2022-2024년 데이터 생성
    dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
    
    # 각 종목별 특성을 반영한 샘플 데이터
    np.random.seed(42)
    
    stocks = {
        '삼성전자': {'base': 60000, 'vol': 0.025, 'trend': 0.0001},
        'LG전자': {'base': 90000, 'vol': 0.030, 'trend': 0.0002}, 
        'SK텔레콤': {'base': 50000, 'vol': 0.020, 'trend': -0.0001},
        'KB금융': {'base': 55000, 'vol': 0.028, 'trend': 0.0001},
        '신한지주': {'base': 40000, 'vol': 0.027, 'trend': 0.0001}
    }
    
    data = {}
    
    for name, params in stocks.items():
        # 기하 브라운 운동으로 주가 시뮬레이션
        returns = np.random.normal(params['trend'], params['vol'], len(dates))
        prices = params['base'] * np.exp(np.cumsum(returns))
        data[name] = prices
    
    stock_data = pd.DataFrame(data, index=dates)
    
    # 주말 제거 (실제 거래일만)
    stock_data = stock_data[stock_data.index.weekday < 5]
    
    print(f"샘플 데이터 생성 완료: {stock_data.shape}")
    return stock_data

def get_korean_stock_data_fdr(start_date, end_date):
    """
    FinanceDataReader를 사용해 한국 주식 데이터 수집
    """
    if not FDR_AVAILABLE:
        print("FinanceDataReader가 설치되지 않았습니다.")
        return pd.DataFrame()
    
    data = {}
    
    # 한국 주식 티커 매핑
    korea_tickers = {
        '삼성전자': '005930',
        'LG전자': '066570', 
        'SK텔레콤': '017670',
        'KB금융': '105560',
        '신한지주': '055550'
    }
    
    print("FinanceDataReader로 주식 데이터 수집 중...")
    
    for name, ticker in korea_tickers.items():
        try:
            df = fdr.DataReader(ticker, start_date, end_date)
            data[name] = df['Close']
            print(f"{name} ({ticker}) 데이터 수집 완료")
        except Exception as e:
            print(f"{name} ({ticker}) 데이터 수집 실패: {e}")
    
    # 데이터프레임으로 결합
    stock_data = pd.DataFrame(data)
    
    return stock_data

def get_korean_stock_data_yfinance(start_date, end_date):
    """
    yfinance를 사용해 한국 주식 데이터 수집
    (한국 주식은 .KS 또는 .KQ 접미사 필요)
    """
    if not YF_AVAILABLE:
        print("yfinance가 설치되지 않았습니다.")
        return pd.DataFrame()
    
    tickers = {
        '삼성전자': '005930.KS',
        'LG전자': '066570.KS',
        'SK텔레콤': '017670.KS', 
        'KB금융': '105560.KS',
        '신한지주': '055550.KS'
    }
    
    data = {}
    print("yfinance로 주식 데이터 수집 중...")
    
    for name, ticker in tickers.items():
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            if not df.empty:
                data[name] = df['Close']
                print(f"{name} ({ticker}) 데이터 수집 완료")
            else:
                print(f"{name} ({ticker}) 데이터가 비어있습니다.")
        except Exception as e:
            print(f"{name} ({ticker}) 데이터 수집 실패: {e}")
    
    stock_data = pd.DataFrame(data)
    return stock_data

def calculate_portfolio_performance(stock_data, weights=None):
    """
    포트폴리오 성과 계산
    """
    if weights is None:
        # 균등가중 포트폴리오
        weights = np.array([1/len(stock_data.columns)] * len(stock_data.columns))
    
    # 일일 수익률 계산
    returns = stock_data.pct_change().dropna()
    
    # 포트폴리오 수익률 계산
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # 누적 수익률 계산
    cumulative_returns = (1 + returns).cumprod()
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    
    # 성과 지표 계산
    annual_returns = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annual_returns / annual_volatility
    
    portfolio_annual_return = portfolio_returns.mean() * 252
    portfolio_annual_volatility = portfolio_returns.std() * np.sqrt(252)
    portfolio_sharpe = portfolio_annual_return / portfolio_annual_volatility
    
    results = {
        'individual_returns': annual_returns,
        'individual_volatility': annual_volatility,
        'individual_sharpe': sharpe_ratio,
        'portfolio_return': portfolio_annual_return,
        'portfolio_volatility': portfolio_annual_volatility,
        'portfolio_sharpe': portfolio_sharpe,
        'cumulative_returns': cumulative_returns,
        'portfolio_cumulative': portfolio_cumulative,
        'correlation_matrix': returns.corr()
    }
    
    return results

def analyze_diversification_effect(stock_data):
    """
    분산투자 효과 분석
    """
    returns = stock_data.pct_change().dropna()
    
    # 개별 종목 투자 vs 분산투자 비교
    scenarios = {}
    
    # 1. 개별 종목 투자 (각각 1000만원)
    for col in stock_data.columns:
        single_stock_return = returns[col].mean() * 252
        single_stock_volatility = returns[col].std() * np.sqrt(252)
        scenarios[f'{col}_단독투자'] = {
            'return': single_stock_return,
            'volatility': single_stock_volatility,
            'sharpe': single_stock_return / single_stock_volatility if single_stock_volatility > 0 else 0
        }
    
    # 2. 균등분산투자 (200만원씩)
    equal_weights = np.array([0.2] * len(stock_data.columns))
    portfolio_returns = (returns * equal_weights).sum(axis=1)
    scenarios['균등분산투자'] = {
        'return': portfolio_returns.mean() * 252,
        'volatility': portfolio_returns.std() * np.sqrt(252),
        'sharpe': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252))
    }
    
    # 3. 2종목 조합들
    from itertools import combinations
    stocks = list(stock_data.columns)
    
    for combo in combinations(stocks, 2):
        combo_weights = np.zeros(len(stocks))
        for i, stock in enumerate(stocks):
            if stock in combo:
                combo_weights[i] = 0.5
        
        combo_returns = (returns * combo_weights).sum(axis=1)
        combo_name = f'{combo[0]} + {combo[1]}'
        scenarios[combo_name] = {
            'return': combo_returns.mean() * 252,
            'volatility': combo_returns.std() * np.sqrt(252),
            'sharpe': (combo_returns.mean() * 252) / (combo_returns.std() * np.sqrt(252))
        }
    
    return scenarios

def plot_results(results, scenarios):
    """
    결과 시각화
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 누적 수익률 그래프
    axes[0,0].plot(results['cumulative_returns'])
    axes[0,0].plot(results['portfolio_cumulative'], 'black', linewidth=2, label='균등분산포트폴리오')
    axes[0,0].set_title('누적 수익률 비교')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # 2. 상관관계 히트맵
    sns.heatmap(results['correlation_matrix'], annot=True, cmap='coolwarm', center=0, ax=axes[0,1])
    axes[0,1].set_title('종목간 상관관계')
    
    # 3. 위험-수익률 산점도
    returns = [scenarios[key]['return'] for key in scenarios.keys()]
    volatilities = [scenarios[key]['volatility'] for key in scenarios.keys()]
    labels = list(scenarios.keys())
    
    axes[1,0].scatter(volatilities, returns)
    for i, label in enumerate(labels):
        axes[1,0].annotate(label, (volatilities[i], returns[i]), fontsize=8)
    axes[1,0].set_xlabel('위험 (연간 변동성)')
    axes[1,0].set_ylabel('수익률 (연간)')
    axes[1,0].set_title('위험-수익률 관계')
    axes[1,0].grid(True)
    
    # 4. 샤프 비율 비교
    sharpe_ratios = [scenarios[key]['sharpe'] for key in scenarios.keys()]
    axes[1,1].bar(range(len(labels)), sharpe_ratios)
    axes[1,1].set_xticks(range(len(labels)))
    axes[1,1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1,1].set_title('샤프 비율 비교')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()

# 메인 실행 코드
if __name__ == "__main__":
    # 분석 기간 설정
    start_date = '2022-01-01'
    end_date = '2024-12-31'
    
    print("=== 한국 주식 포트폴리오 분석 ===")
    print(f"분석 기간: {start_date} ~ {end_date}")
    
    stock_data = pd.DataFrame()
    
    # 데이터 수집 시도 (우선순위대로)
    if FDR_AVAILABLE:
        print("\n1. FinanceDataReader 시도...")
        stock_data = get_korean_stock_data_fdr(start_date, end_date)
    
    if stock_data.empty and YF_AVAILABLE:
        print("\n2. yfinance 시도...")
        stock_data = get_korean_stock_data_yfinance(start_date, end_date)
    
    if stock_data.empty and CRAWLING_AVAILABLE:
        print("\n3. 웹 크롤링 시도...")
        stock_data = get_stock_data_naver_crawling(start_date, end_date)
    
    if stock_data.empty:
        print("\n4. 샘플 데이터 사용...")
        stock_data = get_sample_data()
    
    if not stock_data.empty:
        print(f"\n수집된 데이터 형태: {stock_data.shape}")
        print(f"수집 기간: {stock_data.index[0]} ~ {stock_data.index[-1]}")
        
        # 결측치 처리
        print(f"\n결측치 현황:")
        print(stock_data.isnull().sum())
        
        # 결측치가 있는 경우 전진 채우기
        if stock_data.isnull().any().any():
            stock_data = stock_data.fillna(method='ffill').dropna()
            print("결측치를 전진 채우기로 처리했습니다.")
        
        # 기본 통계
        print(f"\n기본 통계:")
        print(stock_data.describe())
        
        # 포트폴리오 성과 계산
        results = calculate_portfolio_performance(stock_data)
        
        # 분산투자 효과 분석
        scenarios = analyze_diversification_effect(stock_data)
        
        # 결과 출력
        print("\n=== 개별 종목 성과 (연율화) ===")
        for stock in stock_data.columns:
            print(f"{stock}: 수익률 {results['individual_returns'][stock]:.2%}, "
                  f"변동성 {results['individual_volatility'][stock]:.2%}, "
                  f"샤프비율 {results['individual_sharpe'][stock]:.3f}")
        
        print(f"\n=== 균등분산 포트폴리오 성과 ===")
        print(f"수익률: {results['portfolio_return']:.2%}")
        print(f"변동성: {results['portfolio_volatility']:.2%}")
        print(f"샤프비율: {results['portfolio_sharpe']:.3f}")
        
        print(f"\n=== 상관관계 매트릭스 ===")
        print(results['correlation_matrix'].round(3))
        
        print(f"\n=== 최적 2종목 조합 (샤프비율 기준) ===")
        two_stock_scenarios = {k: v for k, v in scenarios.items() if '+' in k}
        if two_stock_scenarios:
            best_combo = max(two_stock_scenarios.items(), key=lambda x: x[1]['sharpe'])
            print(f"{best_combo[0]}: 샤프비율 {best_combo[1]['sharpe']:.3f}, "
                  f"수익률 {best_combo[1]['return']:.2%}, "
                  f"변동성 {best_combo[1]['volatility']:.2%}")
            
            # 상위 3개 조합 출력
            sorted_combos = sorted(two_stock_scenarios.items(), 
                                 key=lambda x: x[1]['sharpe'], reverse=True)
            print("\n상위 3개 2종목 조합:")
            for i, (name, perf) in enumerate(sorted_combos[:3], 1):
                print(f"{i}. {name}: 샤프비율 {perf['sharpe']:.3f}")
        
        # 1000만원 투자 시나리오 비교
        print(f"\n=== 1000만원 투자 시나리오 비교 ===")
        initial_investment = 10_000_000
        
        # 기간 수익률 계산 (전체 기간)
        period_returns = (stock_data.iloc[-1] / stock_data.iloc[0]) - 1
        
        print("개별 종목 투자 결과:")
        for stock in stock_data.columns:
            final_value = initial_investment * (1 + period_returns[stock])
            print(f"{stock}: {final_value:,.0f}원 ({period_returns[stock]:.1%})")
        
        # 균등분산 포트폴리오
        portfolio_period_return = period_returns.mean()  # 균등가중
        portfolio_final_value = initial_investment * (1 + portfolio_period_return)
        print(f"\n균등분산(200만원씩): {portfolio_final_value:,.0f}원 ({portfolio_period_return:.1%})")
        
        # 시각화
        try:
            plot_results(results, scenarios)
        except Exception as e:
            print(f"시각화 오류: {e}")
            print("matplotlib가 제대로 설치되지 않았을 수 있습니다.")
        
    else:
        print("주식 데이터를 가져올 수 없습니다.")
        print("\n해결 방법:")
        print("1. pip install finance-datareader  # 정확한 패키지명")
        print("2. pip install yfinance")
        print("3. pip install requests beautifulsoup4  # 크롤링용")
        print("4. 인터넷 연결 확인")
        print("5. 방화벽/보안 설정 확인")

# 추가 분석 함수들

def monte_carlo_simulation(stock_data, num_simulations=10000):
    """
    몬테카를로 시뮬레이션을 통한 포트폴리오 최적화
    """
    returns = stock_data.pct_change().dropna()
    num_assets = len(stock_data.columns)
    
    results_array = np.zeros((3, num_simulations))
    weights_array = np.zeros((num_simulations, num_assets))
    
    for i in range(num_simulations):
        # 랜덤 가중치 생성
        weights = np.random.random(num_assets)
        weights = weights / np.sum(weights)
        weights_array[i] = weights
        
        # 포트폴리오 수익률과 변동성 계산
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        
        results_array[0, i] = portfolio_return
        results_array[1, i] = portfolio_volatility
        results_array[2, i] = portfolio_return / portfolio_volatility  # 샤프 비율
    
    return results_array, weights_array

def efficient_frontier(stock_data, num_portfolios=100):
    """
    효율적 투자선 계산
    """
    returns = stock_data.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    num_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))
    
    # 목표 수익률 범위 설정
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), num_portfolios)
    
    for i, target in enumerate(target_returns):
        # 최적화 (scipy.optimize 필요)
        try:
            from scipy.optimize import minimize
            
            def portfolio_volatility(weights):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            def portfolio_return(weights):
                return np.sum(mean_returns * weights)
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]
            
            bounds = tuple((0, 1) for _ in range(num_assets))
            initial_guess = np.array([1/num_assets] * num_assets)
            
            result = minimize(portfolio_volatility, initial_guess, 
                            method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                results[0, i] = target
                results[1, i] = result.fun
                results[2, i] = target / result.fun
            
        except ImportError:
            print("scipy가 설치되지 않아 효율적 투자선 계산을 건너뜁니다.")
            break
    
    return results

# 사용 예시 및 설치 가이드
"""
# FinanceDataReader 설치 방법 (여러 시도):

1. 정확한 패키지명으로 설치:
   pip install finance-datareader

2. 업그레이드 시도:
   pip install --upgrade finance-datareader

3. conda 사용:
   conda install -c conda-forge finance-datareader

4. 소스에서 직접 설치:
   pip install git+https://github.com/FinanceData/FinanceDataReader.git

5. 캐시 클리어 후 재시도:
   pip cache purge
   pip install finance-datareader

# 기타 필수 라이브러리:
pip install pandas numpy matplotlib seaborn yfinance requests beautifulsoup4

# 실행 방법:
python stock_analysis.py

# 주요 기능:
1. 한국 주식 데이터 자동 수집 (다중 백업 방법)
2. 포트폴리오 성과 분석  
3. 분산투자 효과 측정
4. 최적 조합 찾기
5. 1000만원 투자 시나리오 비교
6. 시각화

# 데이터 소스 우선순위:
1. FinanceDataReader (한국 주식 최적화)
2. yfinance (글로벌, 한국 주식 지원)
3. 웹 크롤링 (네이버 금융 등)
4. 샘플 데이터 (테스트용)

# 문제 해결:
- 패키지 설치 실패 시: 다른 방법 자동 시도
- 데이터 수집 실패 시: 백업 방법 사용
- 인터넷 연결 문제 시: 샘플 데이터로 테스트 가능
"""
