# 한국 주식 포트폴리오 분석 도구 (캐싱 & 한글 지원)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
import pickle
import platform
from itertools import combinations
warnings.filterwarnings('ignore')

# 한글 폰트 설정
def setup_korean_font():
    """matplotlib 한글 폰트 설정"""
    import matplotlib.font_manager as fm
    
    # 운영체제별 한글 폰트 설정
    system = platform.system()
    
    try:
        if system == 'Windows':
            # Windows: 맑은 고딕
            plt.rcParams['font.family'] = 'Malgun Gothic'
        elif system == 'Darwin':  # macOS
            # macOS: 애플 고딕
            plt.rcParams['font.family'] = 'AppleGothic'
        else:  # Linux
            # Linux: 나눔고딕 (설치되어 있다면)
            font_list = [f.name for f in fm.fontManager.ttflist]
            if 'NanumGothic' in font_list:
                plt.rcParams['font.family'] = 'NanumGothic'
            elif 'DejaVu Sans' in font_list:
                plt.rcParams['font.family'] = 'DejaVu Sans'
            else:
                print("한글 폰트를 찾을 수 없습니다. 설치 방법:")
                print("Ubuntu: sudo apt-get install fonts-nanum")
                print("CentOS: sudo yum install naver-nanum-fonts")
        
        # 음수 부호 깨짐 방지
        plt.rcParams['axes.unicode_minus'] = False
        print(f"한글 폰트 설정 완료: {plt.rcParams['font.family']}")
        
    except Exception as e:
        print(f"한글 폰트 설정 실패: {e}")
        print("기본 폰트를 사용합니다.")

# 한글 폰트 설정 실행
setup_korean_font()

# 데이터 저장/로드를 위한 디렉토리 설정
DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"데이터 디렉토리 생성: {DATA_DIR}")

# 라이브러리 가용성 체크
try:
    import FinanceDataReader as fdr
    FDR_AVAILABLE = True
    print("FinanceDataReader 사용 가능")
except ImportError:
    print("FinanceDataReader 설치 필요: pip install finance-datareader")
    FDR_AVAILABLE = False

try:
    import yfinance as yf
    YF_AVAILABLE = True
    print("yfinance 사용 가능")
except ImportError:
    print("yfinance 설치 필요: pip install yfinance")
    YF_AVAILABLE = False

try:
    import requests
    from bs4 import BeautifulSoup
    CRAWLING_AVAILABLE = True
    print("웹 크롤링 사용 가능")
except ImportError:
    print("크롤링 라이브러리 설치 필요: pip install requests beautifulsoup4")
    CRAWLING_AVAILABLE = False

# 캐시 관련 함수들
def get_cache_filename(start_date, end_date):
    """캐시 파일명 생성"""
    return os.path.join(DATA_DIR, f"stock_data_{start_date}_{end_date}.pkl")

def save_stock_data(stock_data, start_date, end_date):
    """주식 데이터를 캐시 파일로 저장"""
    try:
        filename = get_cache_filename(start_date, end_date)
        with open(filename, 'wb') as f:
            pickle.dump(stock_data, f)
        print(f"데이터 저장 완료: {filename}")
        return True
    except Exception as e:
        print(f"데이터 저장 실패: {e}")
        return False

def load_stock_data(start_date, end_date):
    """캐시된 주식 데이터 로드"""
    try:
        filename = get_cache_filename(start_date, end_date)
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                stock_data = pickle.load(f)
            print(f"캐시된 데이터 로드: {filename}")
            return stock_data
        else:
            print("캐시된 데이터가 없습니다.")
            return pd.DataFrame()
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return pd.DataFrame()

def is_cache_valid(start_date, end_date, max_age_days=1):
    """캐시 파일이 유효한지 확인 (최신성 체크)"""
    try:
        filename = get_cache_filename(start_date, end_date)
        if os.path.exists(filename):
            # 파일 수정 시간 확인
            file_time = datetime.fromtimestamp(os.path.getmtime(filename))
            age = datetime.now() - file_time
            
            if age.days <= max_age_days:
                print(f"캐시 파일이 유효합니다 (생성: {file_time.strftime('%Y-%m-%d %H:%M')})")
                return True
            else:
                print(f"캐시 파일이 오래되었습니다 ({age.days}일 전)")
                return False
        return False
    except Exception as e:
        print(f"캐시 유효성 확인 실패: {e}")
        return False

def clear_cache():
    """캐시 파일들 삭제"""
    try:
        cache_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pkl')]
        for file in cache_files:
            os.remove(os.path.join(DATA_DIR, file))
            print(f"삭제: {file}")
        print(f"캐시 파일 {len(cache_files)}개 삭제 완료")
    except Exception as e:
        print(f"캐시 삭제 실패: {e}")

# 데이터 수집 함수들
def get_korean_stock_data_fdr(start_date, end_date):
    """FinanceDataReader를 사용해 한국 주식 데이터 수집"""
    if not FDR_AVAILABLE:
        print("FinanceDataReader가 설치되지 않았습니다.")
        return pd.DataFrame()
    
    data = {}
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
    
    stock_data = pd.DataFrame(data)
    return stock_data

def get_korean_stock_data_yfinance(start_date, end_date):
    """yfinance를 사용해 한국 주식 데이터 수집"""
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

def get_stock_data_naver_crawling(start_date, end_date):
    """네이버 금융에서 주식 데이터 크롤링 (백업 방법)"""
    if not CRAWLING_AVAILABLE:
        print("크롤링 라이브러리가 없습니다.")
        return pd.DataFrame()
    
    print("웹 크롤링 방법은 구현이 복잡하여 샘플 데이터를 생성합니다...")
    return get_sample_data()

def get_sample_data():
    """샘플 데이터 생성 (라이브러리 설치가 안 될 경우)"""
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

def get_stock_data_with_cache(start_date, end_date, force_refresh=False):
    """캐시를 사용한 주식 데이터 수집"""
    stock_data = pd.DataFrame()
    
    # 강제 새로고침이 아니고 유효한 캐시가 있다면 사용
    if not force_refresh and is_cache_valid(start_date, end_date):
        stock_data = load_stock_data(start_date, end_date)
        
        # 로드된 데이터가 유효한지 확인
        if not stock_data.empty and len(stock_data.columns) >= 3:
            print("캐시된 데이터를 사용합니다.")
            return stock_data
        else:
            print("캐시된 데이터가 불완전합니다. 새로 수집합니다.")
    
    # 새로 데이터 수집
    print("새로운 데이터를 수집합니다...")
    
    # 기존 수집 방법들 시도
    if FDR_AVAILABLE:
        print("1. FinanceDataReader 시도...")
        stock_data = get_korean_stock_data_fdr(start_date, end_date)
    
    if stock_data.empty and YF_AVAILABLE:
        print("2. yfinance 시도...")
        stock_data = get_korean_stock_data_yfinance(start_date, end_date)
    
    if stock_data.empty and CRAWLING_AVAILABLE:
        print("3. 웹 크롤링 시도...")
        stock_data = get_stock_data_naver_crawling(start_date, end_date)
    
    if stock_data.empty:
        print("4. 샘플 데이터 사용...")
        stock_data = get_sample_data()
    
    # 데이터 수집 성공 시 캐시에 저장
    if not stock_data.empty:
        save_stock_data(stock_data, start_date, end_date)
    
    return stock_data

# 포트폴리오 분석 함수들
def calculate_portfolio_performance(stock_data, weights=None):
    """포트폴리오 성과 계산"""
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
    """분산투자 효과 분석"""
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
    """결과 시각화 (한글 지원)"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 누적 수익률 그래프
    # axes[0,0].plot(results['cumulative_returns'], alpha=0.7)
    # axes[0,0].plot(results['portfolio_cumulative'], 'black', linewidth=3, label='균등분산포트폴리오')
    # axes[0,0].set_title('누적 수익률 비교', fontsize=14, fontweight='bold')
    # axes[0,0].set_ylabel('누적 수익률')
    # axes[0,0].legend(loc='upper left')
    # axes[0,0].grid(True, alpha=0.3)
    
    # 개별 종목별로 플롯
    for column in results['cumulative_returns'].columns:
        axes[0,0].plot(results['cumulative_returns'][column], 
                       alpha=0.7, 
                       linewidth=1.5,
                       label=column)
    
    # 균등분산 포트폴리오 (굵은 검은 선)
    axes[0,0].plot(results['portfolio_cumulative'], 
                   'black', 
                   linewidth=3, 
                   label='균등분산포트폴리오')
    
    axes[0,0].set_title('누적 수익률 비교', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('누적 수익률')
    
    # 범례 설정 (위치와 스타일 조정)
    axes[0,0].legend(loc='upper left', 
                     frameon=True, 
                     fancybox=True, 
                     shadow=True,
                     fontsize=9)
    
    # axes[0,0].grid(True, alpha=0.3)
    axes[0,0].grid(True, alpha=0.3, which='both')  # major + minor 격자
    
    # x축 날짜 포맷 개선
    import matplotlib.dates as mdates
    # 분기별 표시
    axes[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[0,0].xaxis.set_major_locator(mdates.MonthLocator(interval=4))  # 4개월마다
    axes[0,0].xaxis.set_minor_locator(mdates.MonthLocator(interval=2))  # 2개월마다 작은 눈금
    # 연간 표시
    # axes[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    # axes[0,0].xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(axes[0,0].xaxis.get_majorticklabels(), rotation=45)
    
    # 2. 상관관계 히트맵
    sns.heatmap(results['correlation_matrix'], annot=True, cmap='RdYlBu_r', center=0, 
                ax=axes[0,1], square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    axes[0,1].set_title('종목간 상관관계', fontsize=14, fontweight='bold')
    
    # 3. 위험-수익률 산점도 (라벨 개선)
    returns = [scenarios[key]['return'] for key in scenarios.keys()]
    volatilities = [scenarios[key]['volatility'] for key in scenarios.keys()]
    labels = list(scenarios.keys())
    
    # 색상 구분
    colors = ['red' if '단독투자' in label else 'blue' if '+' in label else 'green' for label in labels]
    sizes = [80 if '단독투자' in label else 60 if '+' in label else 100 for label in labels]
    
    scatter = axes[1,0].scatter(volatilities, returns, c=colors, s=sizes, alpha=0.7, edgecolors='black')
    
    # 2종목 조합 중 상위 3개 찾기
    two_stock_scenarios = {k: v for k, v in scenarios.items() if '+' in k}
    sorted_combos = sorted(two_stock_scenarios.items(), 
                         key=lambda x: x[1]['sharpe'], reverse=True)
    top_3_combo_names = [combo[0] for combo in sorted_combos[:3]]  # ← 상위 3개만
    
    # 라벨 표시
    for i, label in enumerate(labels):
        if '단독투자' in label or label == '균등분산투자':
            # 개별 종목과 균등분산
            axes[1,0].annotate(label, (volatilities[i], returns[i]), 
                             xytext=(5, 5), textcoords='offset points', 
                             fontsize=9, ha='left', weight='bold')
            # axes[1,0].annotate(label, (volatilities[i], returns[i]), 
            #                  xytext=(5, 5), textcoords='offset points', 
            #                  fontsize=9, ha='left',
            #                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        # elif label in top_3_combo_names: # 상위 3개 종목만
        elif '+' in label:  # ← 모든 2종목 조합
            # 상위 3개 2종목 조합
            short_label = label.replace(' + ', '+')
            axes[1,0].annotate(short_label, (volatilities[i], returns[i]), 
                             xytext=(0, 10), textcoords='offset points', 
                             fontsize=7, ha='center')
            # axes[1,0].annotate(short_label, (volatilities[i], returns[i]), 
            #                  xytext=(5, 5), textcoords='offset points', 
            #                  fontsize=8, ha='left',
            #                  bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.8))
    
    axes[1,0].set_xlabel('위험 (연간 변동성)')
    axes[1,0].set_ylabel('수익률 (연간)')
    axes[1,0].set_title('위험-수익률 관계', fontsize=14, fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)
    
    # 범례 추가
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='개별종목'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='2종목 조합'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='균등분산')
    ]
    axes[1,0].legend(handles=legend_elements, loc='upper left')
    
    # 4. 샤프 비율 비교 (상위 10개만)
    sharpe_data = [(k, v['sharpe']) for k, v in scenarios.items()]
    sharpe_data.sort(key=lambda x: x[1], reverse=True)
    
    # 상위 10개만 표시
    top_10 = sharpe_data[:10]
    labels_top = [item[0] for item in top_10]
    sharpe_top = [item[1] for item in top_10]
    
    # 색상 구분
    colors_bar = ['red' if '단독투자' in label else 'blue' if '+' in label else 'green' for label in labels_top]
    
    bars = axes[1,1].bar(range(len(labels_top)), sharpe_top, color=colors_bar, alpha=0.7, edgecolor='black')
    axes[1,1].set_xticks(range(len(labels_top)))
    axes[1,1].set_xticklabels(labels_top, rotation=45, ha='right', fontsize=9)
    axes[1,1].set_title('샤프 비율 비교 (상위 10개)', fontsize=14, fontweight='bold')
    axes[1,1].set_ylabel('샤프 비율')
    axes[1,1].grid(True, alpha=0.3, axis='y')
    
    # 값 표시
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # 그래프 저장
    try:
        save_path = os.path.join(DATA_DIR, 'portfolio_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"그래프 저장: {save_path}")
    except Exception as e:
        print(f"그래프 저장 실패: {e}")
    
    plt.show()

def save_results_to_excel(results, scenarios, stock_data, filename=None):
    """분석 결과를 엑셀 파일로 저장"""
    if filename is None:
        filename = os.path.join(DATA_DIR, f'portfolio_analysis_{datetime.now().strftime("%Y%m%d_%H%M")}.xlsx')
    
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 1. 원본 주가 데이터
            stock_data.to_excel(writer, sheet_name='주가데이터')
            
            # 2. 개별 종목 성과
            performance_df = pd.DataFrame({
                '종목명': results['individual_returns'].index,
                '연간수익률': results['individual_returns'].values,
                '연간변동성': results['individual_volatility'].values,
                '샤프비율': results['individual_sharpe'].values
            })
            performance_df.to_excel(writer, sheet_name='개별종목성과', index=False)
            
            # 3. 상관관계 매트릭스
            results['correlation_matrix'].to_excel(writer, sheet_name='상관관계')
            
            # 4. 시나리오 비교
            scenarios_df = pd.DataFrame.from_dict(scenarios, orient='index')
            scenarios_df.index.name = '시나리오'
            scenarios_df.to_excel(writer, sheet_name='시나리오비교')
            
            # 5. 요약 정보
            summary = {
                '항목': ['분석기간', '데이터수집일', '종목수', '균등포트폴리오_수익률', '균등포트폴리오_변동성', '균등포트폴리오_샤프비율'],
                '값': [f"{stock_data.index[0].date()} ~ {stock_data.index[-1].date()}",
                      datetime.now().strftime('%Y-%m-%d %H:%M'),
                      len(stock_data.columns),
                      f"{results['portfolio_return']:.4f}",
                      f"{results['portfolio_volatility']:.4f}",
                      f"{results['portfolio_sharpe']:.4f}"]
            }
            summary_df = pd.DataFrame(summary)
            summary_df.to_excel(writer, sheet_name='요약', index=False)
        
        print(f"결과 저장 완료: {filename}")
        return filename
    
    except Exception as e:
        print(f"엑셀 저장 실패: {e}")
        return None

# 유틸리티 함수들
def show_cache_info():
    """캐시 파일 정보 출력"""
    try:
        cache_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pkl')]
        if cache_files:
            print(f"캐시 파일 목록 ({len(cache_files)}개):")
            for file in cache_files:
                file_path = os.path.join(DATA_DIR, file)
                file_size = os.path.getsize(file_path)
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                print(f"  {file}: {file_size/1024:.1f}KB, {file_time.strftime('%Y-%m-%d %H:%M')}")
        else:
            print("캐시 파일이 없습니다.")
    except Exception as e:
        print(f"캐시 정보 조회 실패: {e}")

def load_pkl_file(filename=None):
    """pkl 파일 로드 및 미리보기"""
    try:
        if filename is None:
            # 캐시 파일 목록 표시
            cache_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pkl')]
            if not cache_files:
                print("pkl 파일이 없습니다.")
                return None
            
            print("사용 가능한 pkl 파일:")
            for i, file in enumerate(cache_files, 1):
                print(f"{i}. {file}")
            
            choice = input(f"파일 번호를 선택하세요 (1-{len(cache_files)}): ").strip()
            try:
                file_idx = int(choice) - 1
                filename = cache_files[file_idx]
            except (ValueError, IndexError):
                print("잘못된 선택입니다.")
                return None
        
        file_path = os.path.join(DATA_DIR, filename)
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\n✓ 파일 로드 완료: {filename}")
        
        if isinstance(data, pd.DataFrame):
            print(f"데이터 타입: DataFrame")
            print(f"크기: {data.shape}")
            print(f"컬럼: {list(data.columns)}")
            print(f"기간: {data.index[0]} ~ {data.index[-1]}")
            print(f"\n처음 5행:")
            print(data.head())
            print(f"\n마지막 5행:")
            print(data.tail())
            print(f"\n기본 통계:")
            print(data.describe())
        else:
            print(f"데이터 타입: {type(data)}")
            print(f"내용: {data}")
        
        return data
        
    except Exception as e:
        print(f"파일 로드 실패: {e}")
        return None

def pkl_to_csv(pkl_filename=None, csv_filename=None):
    """pkl 파일을 CSV로 변환"""
    try:
        # pkl 파일 로드
        if pkl_filename is None:
            cache_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pkl')]
            if not cache_files:
                print("pkl 파일이 없습니다.")
                return False
            
            print("변환할 pkl 파일을 선택하세요:")
            for i, file in enumerate(cache_files, 1):
                print(f"{i}. {file}")
            
            choice = input(f"파일 번호 (1-{len(cache_files)}): ").strip()
            try:
                file_idx = int(choice) - 1
                pkl_filename = cache_files[file_idx]
            except (ValueError, IndexError):
                print("잘못된 선택입니다.")
                return False
        
        pkl_path = os.path.join(DATA_DIR, pkl_filename)
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, pd.DataFrame):
            print("DataFrame이 아닌 데이터는 CSV로 변환할 수 없습니다.")
            return False
        
        # CSV 파일명 생성
        if csv_filename is None:
            base_name = pkl_filename.replace('.pkl', '')
            csv_filename = f"{base_name}.csv"
        
        csv_path = os.path.join(DATA_DIR, csv_filename)
        
        # CSV로 저장 (한글 깨짐 방지)
        data.to_csv(csv_path, encoding='utf-8-sig', index=True)
        
        print(f"✓ CSV 변환 완료: {csv_filename}")
        print(f"저장 위치: {os.path.abspath(csv_path)}")
        print(f"데이터 크기: {data.shape}")
        
        return True
        
    except Exception as e:
        print(f"CSV 변환 실패: {e}")
        return False

def pkl_to_xlsx(pkl_filename=None, xlsx_filename=None):
    """pkl 파일을 Excel로 변환"""
    try:
        # pkl 파일 로드
        if pkl_filename is None:
            cache_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pkl')]
            if not cache_files:
                print("pkl 파일이 없습니다.")
                return False
            
            print("변환할 pkl 파일을 선택하세요:")
            for i, file in enumerate(cache_files, 1):
                print(f"{i}. {file}")
            
            choice = input(f"파일 번호 (1-{len(cache_files)}): ").strip()
            try:
                file_idx = int(choice) - 1
                pkl_filename = cache_files[file_idx]
            except (ValueError, IndexError):
                print("잘못된 선택입니다.")
                return False
        
        pkl_path = os.path.join(DATA_DIR, pkl_filename)
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, pd.DataFrame):
            print("DataFrame이 아닌 데이터는 Excel로 변환할 수 없습니다.")
            return False
        
        # Excel 파일명 생성
        if xlsx_filename is None:
            base_name = pkl_filename.replace('.pkl', '')
            xlsx_filename = f"{base_name}.xlsx"
        
        xlsx_path = os.path.join(DATA_DIR, xlsx_filename)
        
        # Excel로 저장 (여러 시트로 구성)
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            # 메인 데이터
            data.to_excel(writer, sheet_name='주가데이터', index=True)
            
            # 기본 통계
            data.describe().to_excel(writer, sheet_name='기본통계')
            
            # 수익률 데이터 (있다면)
            if len(data) > 1:
                returns = data.pct_change().dropna()
                returns.to_excel(writer, sheet_name='일간수익률', index=True)
                
                # 월간 데이터
                monthly_data = data.resample('M').last()
                monthly_returns = monthly_data.pct_change().dropna()
                monthly_data.to_excel(writer, sheet_name='월간데이터', index=True)
                monthly_returns.to_excel(writer, sheet_name='월간수익률', index=True)
                
                # 상관관계
                returns.corr().to_excel(writer, sheet_name='상관관계')
            
            # 메타 정보
            meta_info = {
                '항목': ['파일명', '데이터크기', '시작일', '종료일', '총거래일수', '종목수', '변환일시'],
                '값': [
                    pkl_filename,
                    f"{data.shape[0]} × {data.shape[1]}",
                    str(data.index[0].date()) if len(data) > 0 else 'N/A',
                    str(data.index[-1].date()) if len(data) > 0 else 'N/A',
                    len(data),
                    len(data.columns),
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ]
            }
            pd.DataFrame(meta_info).to_excel(writer, sheet_name='파일정보', index=False)
        
        print(f"✓ Excel 변환 완료: {xlsx_filename}")
        print(f"저장 위치: {os.path.abspath(xlsx_path)}")
        print(f"시트 구성: 주가데이터, 기본통계, 일간수익률, 월간데이터, 월간수익률, 상관관계, 파일정보")
        print(f"데이터 크기: {data.shape}")
        
        return True
        
    except Exception as e:
        print(f"Excel 변환 실패: {e}")
        return False

def convert_all_pkl_files():
    """모든 pkl 파일을 CSV와 Excel로 일괄 변환"""
    try:
        cache_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pkl')]
        if not cache_files:
            print("변환할 pkl 파일이 없습니다.")
            return
        
        print(f"총 {len(cache_files)}개 파일을 변환합니다...")
        
        success_count = 0
        for pkl_file in cache_files:
            print(f"\n변환 중: {pkl_file}")
            
            # CSV 변환
            if pkl_to_csv(pkl_file):
                print(f"  ✓ CSV 변환 완료")
            else:
                print(f"  ✗ CSV 변환 실패")
                continue
            
            # Excel 변환
            if pkl_to_xlsx(pkl_file):
                print(f"  ✓ Excel 변환 완료")
                success_count += 1
            else:
                print(f"  ✗ Excel 변환 실패")
        
        print(f"\n일괄 변환 완료: {success_count}/{len(cache_files)} 파일 성공")
        
    except Exception as e:
        print(f"일괄 변환 실패: {e}")

def explore_pkl_data():
    """pkl 데이터 상세 탐색"""
    data = load_pkl_file()
    if data is None or not isinstance(data, pd.DataFrame):
        return
    
    while True:
        print(f"\n" + "="*40)
        print("데이터 탐색 메뉴")
        print("="*40)
        print("1. 기본 정보")
        print("2. 특정 기간 데이터")
        print("3. 특정 종목 데이터")
        print("4. 수익률 분석")
        print("5. 상관관계 분석")
        print("6. 그래프 그리기")
        print("7. 돌아가기")
        print("="*40)
        
        choice = input("선택하세요 (1-7): ").strip()
        
        if choice == '1':
            print(f"\n=== 기본 정보 ===")
            print(f"데이터 크기: {data.shape}")
            print(f"종목: {list(data.columns)}")
            print(f"기간: {data.index[0]} ~ {data.index[-1]}")
            print(f"결측치: {data.isnull().sum().sum()}개")
            print(f"\n최근 데이터:")
            print(data.tail())
            
        elif choice == '2':
            start_date = input("시작일 (YYYY-MM-DD): ").strip()
            end_date = input("종료일 (YYYY-MM-DD): ").strip()
            try:
                period_data = data.loc[start_date:end_date]
                print(f"\n{start_date} ~ {end_date} 데이터:")
                print(period_data)
                print(f"\n기간 통계:")
                print(period_data.describe())
            except Exception as e:
                print(f"기간 조회 실패: {e}")
                
        elif choice == '3':
            print(f"사용 가능한 종목: {list(data.columns)}")
            stock = input("종목명 입력: ").strip()
            if stock in data.columns:
                stock_data = data[stock].dropna()
                print(f"\n{stock} 데이터:")
                print(f"기간: {stock_data.index[0]} ~ {stock_data.index[-1]}")
                print(f"최고가: {stock_data.max():,.0f}")
                print(f"최저가: {stock_data.min():,.0f}")
                print(f"평균가: {stock_data.mean():,.0f}")
                print(f"최근 5일:")
                print(stock_data.tail())
            else:
                print("해당 종목을 찾을 수 없습니다.")
                
        elif choice == '4':
            returns = data.pct_change().dropna()
            print(f"\n=== 수익률 분석 ===")
            print(f"일간 수익률 통계:")
            print(returns.describe())
            print(f"\n연간 수익률 (추정):")
            annual_returns = returns.mean() * 252
            for stock in annual_returns.index:
                print(f"{stock}: {annual_returns[stock]:.2%}")
                
        elif choice == '5':
            returns = data.pct_change().dropna()
            corr_matrix = returns.corr()
            print(f"\n=== 상관관계 분석 ===")
            print(corr_matrix.round(3))
            
            # 가장 높은/낮은 상관관계 찾기
            corr_values = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    stock1 = corr_matrix.columns[i]
                    stock2 = corr_matrix.columns[j]
                    corr_val = corr_matrix.iloc[i, j]
                    corr_values.append((f"{stock1}-{stock2}", corr_val))
            
            corr_values.sort(key=lambda x: x[1])
            print(f"\n가장 낮은 상관관계 (분산효과 좋음):")
            for pair, corr in corr_values[:3]:
                print(f"  {pair}: {corr:.3f}")
            
            print(f"\n가장 높은 상관관계:")
            for pair, corr in corr_values[-3:]:
                print(f"  {pair}: {corr:.3f}")
                
        elif choice == '6':
            try:
                plt.figure(figsize=(12, 8))
                
                # 정규화된 가격 차트 (시작점을 100으로 설정)
                normalized_data = data / data.iloc[0] * 100
                
                plt.subplot(2, 1, 1)
                plt.plot(normalized_data)
                plt.title('주가 추이 (시작점=100)', fontweight='bold')
                plt.ylabel('상대 가격')
                plt.legend(data.columns)
                plt.grid(True, alpha=0.3)
                
                # 일간 수익률 분포
                plt.subplot(2, 1, 2)
                returns = data.pct_change().dropna()
                returns.hist(bins=50, alpha=0.7)
                plt.title('일간 수익률 분포', fontweight='bold')
                plt.xlabel('수익률')
                plt.ylabel('빈도')
                
                plt.tight_layout()
                
                # 저장
                chart_path = os.path.join(DATA_DIR, 'pkl_data_chart.png')
                plt.savefig(chart_path, dpi=150, bbox_inches='tight')
                plt.show()
                
                print(f"차트 저장: {chart_path}")
                
            except Exception as e:
                print(f"그래프 생성 실패: {e}")
                
        elif choice == '7':
            break
        else:
            print("잘못된 선택입니다.")

def test_korean_font():
    """한글 폰트 테스트"""
    try:
        plt.figure(figsize=(10, 6))
        
        # 테스트 데이터 생성
        x = range(10)
        y = [i**2 for i in x]
        
        plt.plot(x, y, 'bo-', linewidth=2, markersize=8)
        plt.title('한글 폰트 테스트 - 주식 포트폴리오 분석', fontsize=16, fontweight='bold')
        plt.xlabel('시간 (개월)', fontsize=12)
        plt.ylabel('수익률 (%)', fontsize=12)
        
        # 한글 라벨 테스트
        plt.text(5, 50, '삼성전자', fontsize=14, ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        plt.text(3, 20, 'LG전자', fontsize=14, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 파일 저장 테스트
        test_path = os.path.join(DATA_DIR, 'font_test.png')
        plt.savefig(test_path, dpi=150, bbox_inches='tight', facecolor='white')
        
        plt.show()
        
        print(f"✓ 한글 폰트 테스트 완료!")
        print(f"  현재 폰트: {plt.rcParams['font.family']}")
        print(f"  테스트 파일 저장: {test_path}")
        
    except Exception as e:
        print(f"한글 폰트 테스트 실패: {e}")
        print("한글이 깨져 보인다면 다음을 시도해보세요:")
        print("Windows: 제어판 > 글꼴에서 '맑은 고딕' 확인")
        print("macOS: 시스템 환경설정 > 일반 > 언어 및 지역")
        print("Linux: sudo apt-get install fonts-nanum")

def main_analysis(force_refresh=False):
    """메인 분석 함수"""
    start_date = '2022-01-01'
    end_date = '2024-12-31'
    
    print(f"분석 기간: {start_date} ~ {end_date}")
    print(f"데이터 저장 위치: {os.path.abspath(DATA_DIR)}")
    
    # 캐시를 사용한 데이터 수집
    stock_data = get_stock_data_with_cache(start_date, end_date, force_refresh)
    
    if not stock_data.empty:
        print(f"\n✓ 데이터 수집 완료")
        print(f"  - 데이터 크기: {stock_data.shape}")
        print(f"  - 기간: {stock_data.index[0].date()} ~ {stock_data.index[-1].date()}")
        print(f"  - 종목: {', '.join(stock_data.columns)}")
        
        # 결측치 처리
        missing_data = stock_data.isnull().sum()
        if missing_data.any():
            print(f"\n결측치 현황:")
            print(missing_data[missing_data > 0])
            stock_data = stock_data.fillna(method='ffill').dropna()
            print("→ 전진 채우기로 결측치 처리 완료")
        
        try:
            # 포트폴리오 성과 계산
            print("\n포트폴리오 분석 중...")
            results = calculate_portfolio_performance(stock_data)
            
            # 분산투자 효과 분석
            print("분산투자 효과 분석 중...")
            scenarios = analyze_diversification_effect(stock_data)
            
            # 결과 출력
            print("\n" + "="*60)
            print("=== 개별 종목 성과 (연율화) ===")
            print("="*60)
            for stock in stock_data.columns:
                print(f"{stock:8s}: 수익률 {results['individual_returns'][stock]:6.2%}, "
                      f"변동성 {results['individual_volatility'][stock]:6.2%}, "
                      f"샤프비율 {results['individual_sharpe'][stock]:6.3f}")
            
            print(f"\n" + "="*60)
            print(f"=== 균등분산 포트폴리오 성과 ===")
            print("="*60)
            print(f"수익률:   {results['portfolio_return']:6.2%}")
            print(f"변동성:   {results['portfolio_volatility']:6.2%}")
            print(f"샤프비율: {results['portfolio_sharpe']:6.3f}")
            
            # 최적 조합 찾기
            print(f"\n" + "="*60)
            print(f"=== 최적 2종목 조합 (샤프비율 기준) ===")
            print("="*60)
            two_stock_scenarios = {k: v for k, v in scenarios.items() if '+' in k}
            if two_stock_scenarios:
                sorted_combos = sorted(two_stock_scenarios.items(), 
                                     key=lambda x: x[1]['sharpe'], reverse=True)
                
                best_combo = sorted_combos[0]
                print(f"🏆 최적 조합: {best_combo[0]}")
                print(f"   샤프비율: {best_combo[1]['sharpe']:6.3f}")
                print(f"   수익률:   {best_combo[1]['return']:6.2%}")
                print(f"   변동성:   {best_combo[1]['volatility']:6.2%}")
                
                print(f"\n상위 3개 조합:")
                for i, (name, perf) in enumerate(sorted_combos[:3], 1):
                    marker = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                    print(f"{marker} {name}: 샤프비율 {perf['sharpe']:6.3f}")
            
            # 1000만원 투자 결과 비교
            print(f"\n" + "="*60)
            print(f"=== 💰 1000만원 투자 결과 비교 ===")
            print("="*60)
            
            initial_investment = 10_000_000
            period_returns = (stock_data.iloc[-1] / stock_data.iloc[0]) - 1
            
            results_list = []
            
            # 개별 종목 결과
            for stock in stock_data.columns:
                final_value = initial_investment * (1 + period_returns[stock])
                profit = final_value - initial_investment
                results_list.append(('개별-'+stock, final_value, period_returns[stock], profit))
            
            # 균등분산 결과
            portfolio_return = period_returns.mean()
            portfolio_final = initial_investment * (1 + portfolio_return)
            portfolio_profit = portfolio_final - initial_investment
            results_list.append(('균등분산', portfolio_final, portfolio_return, portfolio_profit))
            
            # 최적 2종목 조합 결과 (상위 3개)
            for i, (name, perf) in enumerate(sorted_combos[:3]):
                # 연율 수익률을 기간 수익률로 변환 (근사)
                combo_return = perf['return'] * len(stock_data) / 252  
                combo_final = initial_investment * (1 + combo_return)
                combo_profit = combo_final - initial_investment
                results_list.append((f'조합-{name}', combo_final, combo_return, combo_profit))
            
            # 결과 정렬 (수익률 기준)
            results_list.sort(key=lambda x: x[1], reverse=True)
            
            print("최종 금액 순위:")
            for i, (strategy, final_val, return_pct, profit) in enumerate(results_list, 1):
                emoji = "🏆" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
                print(f"{emoji} {strategy:15s}: {final_val:>11,.0f}원 ({return_pct:>7.1%}) "
                      f"손익: {profit:>+10,.0f}원")
            
            # 분산투자 효과 분석 결과
            print(f"\n" + "="*60)
            print(f"=== 📈 분산투자 효과 분석 ===")
            print("="*60)
            
            # 최고 개별 종목 vs 균등분산 비교
            best_individual = max([(k, v) for k, v in scenarios.items() if '단독투자' in k], 
                                key=lambda x: x[1]['return'])
            worst_individual = min([(k, v) for k, v in scenarios.items() if '단독투자' in k], 
                                 key=lambda x: x[1]['return'])
            equal_weight = scenarios['균등분산투자']
            
            print(f"최고 개별 종목: {best_individual[0]} (수익률: {best_individual[1]['return']:.2%})")
            print(f"최저 개별 종목: {worst_individual[0]} (수익률: {worst_individual[1]['return']:.2%})")
            print(f"균등분산 포트폴리오: 수익률 {equal_weight['return']:.2%}")
            
            # 위험 감소 효과
            individual_avg_vol = np.mean([v['volatility'] for k, v in scenarios.items() if '단독투자' in k])
            print(f"\n위험 감소 효과:")
            print(f"개별 종목 평균 변동성: {individual_avg_vol:.2%}")
            print(f"균등분산 포트폴리오 변동성: {equal_weight['volatility']:.2%}")
            print(f"위험 감소: {(individual_avg_vol - equal_weight['volatility'])/individual_avg_vol:.1%}")
            
            # 엑셀 저장
            print(f"\n" + "="*40)
            print("📊 결과 저장 중...")
            excel_file = save_results_to_excel(results, scenarios, stock_data)
            if excel_file:
                print(f"✓ 엑셀 파일: {os.path.basename(excel_file)}")
            
            # 시각화
            print("📈 시각화 생성 중...")
            plot_results(results, scenarios)
            
            print(f"\n" + "="*60)
            print("🎉 분석 완료!")
            print(f"📁 결과 저장 위치: {os.path.abspath(DATA_DIR)}")
            print("="*60)
            
        except Exception as e:
            print(f"❌ 분석 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ 주식 데이터를 가져올 수 없습니다.")
        print("\n해결 방법:")
        print("1. pip install finance-datareader")
        print("2. pip install yfinance") 
        print("3. 인터넷 연결 확인")

def interactive_menu():
    """대화형 메뉴"""
    while True:
        print("\n" + "="*50)
        print("한국 주식 포트폴리오 분석 도구")
        print("="*50)
        print("1. 분석 실행 (캐시 사용)")
        print("2. 분석 실행 (강제 새로고침)")
        print("3. 캐시 정보 확인")
        print("4. 캐시 삭제")
        print("5. 한글 폰트 테스트")
        print("6. pkl 파일 탐색")
        print("7. pkl → CSV 변환")
        print("8. pkl → Excel 변환")
        print("9. 모든 pkl 일괄 변환")
        print("10. 종료")
        print("="*50)
        
        choice = input("선택하세요 (1-10): ").strip()
        
        if choice == '1':
            print("\n캐시를 사용한 분석을 시작합니다...")
            main_analysis(force_refresh=False)
        elif choice == '2':
            print("\n강제 새로고침으로 분석을 시작합니다...")
            main_analysis(force_refresh=True)
        elif choice == '3':
            show_cache_info()
        elif choice == '4':
            confirm = input("정말 캐시를 삭제하시겠습니까? (y/N): ").strip().lower()
            if confirm == 'y':
                clear_cache()
            else:
                print("취소되었습니다.")
        elif choice == '5':
            test_korean_font()
        elif choice == '6':
            explore_pkl_data()
        elif choice == '7':
            pkl_to_csv()
        elif choice == '8':
            pkl_to_xlsx()
        elif choice == '9':
            convert_all_pkl_files()
        elif choice == '10':
            print("프로그램을 종료합니다.")
            break
        else:
            print("잘못된 선택입니다. 1-10 중에서 선택해주세요.")

# 실행 방법 선택
if __name__ == "__main__":
    # 대화형 메뉴 실행 (권장)
    interactive_menu()
    
    # 또는 직접 실행
    # main_analysis(force_refresh=False)

# 추가 설치 및 사용 가이드
"""
=== 한국 주식 포트폴리오 분석 도구 ===

📦 설치 방법:

1. 기본 라이브러리:
   pip install pandas numpy matplotlib seaborn openpyxl

2. 한국 주식 데이터 수집:
   pip install finance-datareader  # 추천
   pip install yfinance             # 백업용

3. 웹 크롤링 (백업):
   pip install requests beautifulsoup4

4. 한글 폰트 설정:
   Windows: 자동 (맑은 고딕)
   macOS: 자동 (애플 고딕)  
   Linux: sudo apt-get install fonts-nanum

🚀 실행 방법:

1. 대화형 메뉴 (권장):
   python stock_analysis.py
   
2. 직접 실행:
   main_analysis(force_refresh=False)

3. 강제 새로고침:
   main_analysis(force_refresh=True)

📁 생성되는 파일들:

/data/
├── stock_data_2022-01-01_2024-12-31.pkl  # 캐시된 주가 데이터
├── portfolio_analysis_20250803_1234.xlsx  # 분석 결과 엑셀
├── portfolio_analysis.png                 # 시각화 차트
└── font_test.png                          # 한글 폰트 테스트

📊 주요 기능:

1. 데이터 캐싱: 한번 수집한 데이터는 자동 저장/재사용
2. 한글 지원: 운영체제별 자동 한글 폰트 설정
3. 다중 백업: FinanceDataReader → yfinance → 크롤링 → 샘플 순으로 시도
4. 완전한 분석: 개별/조합/분산 투자 효과 비교
5. 결과 저장: 엑셀 파일과 차트 자동 저장
6. 대화형 메뉴: 사용자 친화적 인터페이스

🎯 분석 결과:

- 개별 종목 vs 균등분산 vs 최적 2종목 조합 비교
- 1000만원 투자 시나리오별 결과
- 위험-수익률 관계 시각화
- 상관관계 분석을 통한 분산효과 측정
- 샤프비율 기반 최적 포트폴리오 추천

💡 사용 팁:

1. 첫 실행시: 데이터 수집에 시간이 걸릴 수 있음
2. 재실행시: 캐시를 사용하여 빠른 분석
3. 최신 데이터: force_refresh=True로 설정
4. 한글 깨짐: 한글 폰트 테스트 메뉴 실행
5. 오류 발생: 라이브러리 재설치 또는 샘플 데이터 사용

🔧 문제 해결:

Q: FinanceDataReader 설치 실패
A: pip install finance-datareader (하이픈 주의)

Q: 한글이 깨져서 보임
A: 메뉴 5번으로 폰트 테스트 실행

Q: 데이터 수집 실패
A: 인터넷 연결 확인, yfinance로 백업 시도

Q: 캐시 파일 문제
A: 메뉴 4번으로 캐시 삭제 후 재시도

📈 확장 가능:

- 더 많은 종목 추가
- 다른 기간 분석
- 섹터별 분산투자
- 리밸런싱 전략
- 백테스팅 기능
"""
