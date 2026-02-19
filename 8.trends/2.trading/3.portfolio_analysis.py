# 포트폴리오 최적화 엔진
# 효율적 경계선, 샤프비율 최적화, 몬테카를로 시뮬레이션

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
import platform
import matplotlib.font_manager as fm

def setup_korean_font():
    """matplotlib 한글 폰트 설정"""
    system = platform.system()
    try:
        if system == 'Windows':
            plt.rcParams['font.family'] = 'Malgun Gothic'
        elif system == 'Darwin':  # macOS
            plt.rcParams['font.family'] = 'AppleGothic'
        else:  # Linux
            font_list = [f.name for f in fm.fontManager.ttflist]
            if 'NanumGothic' in font_list:
                plt.rcParams['font.family'] = 'NanumGothic'
            elif 'DejaVu Sans' in font_list:
                plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        print(f"한글 폰트 설정 완료: {plt.rcParams['font.family']}")
    except Exception as e:
        print(f"한글 폰트 설정 실패: {e}")

setup_korean_font()

class PortfolioOptimizer:
    """포트폴리오 최적화 클래스"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.stock_data = None
        self.returns = None
        self.mu = None  # 기대수익률 벡터
        self.sigma = None  # 공분산 행렬
        self.n_assets = 0
        self.asset_names = []
        
    def load_data(self, filename=None):
        """데이터 로드 (pkl, csv, xlsx 지원)"""
        try:
            if filename is None:
                # 사용 가능한 파일 목록 표시
                files = []
                for ext in ['.pkl', '.csv', '.xlsx']:
                    files.extend([f for f in os.listdir(self.data_dir) if f.endswith(ext)])
                
                if not files:
                    print("데이터 파일이 없습니다.")
                    return False
                
                print("사용 가능한 데이터 파일:")
                for i, file in enumerate(files, 1):
                    print(f"{i}. {file}")
                
                choice = input(f"파일 번호를 선택하세요 (1-{len(files)}): ").strip()
                try:
                    file_idx = int(choice) - 1
                    filename = files[file_idx]
                except (ValueError, IndexError):
                    print("잘못된 선택입니다.")
                    return False
            
            file_path = os.path.join(self.data_dir, filename)
            
            # 파일 형식에 따라 로드
            if filename.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    self.stock_data = pickle.load(f)
            elif filename.endswith('.csv'):
                self.stock_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            elif filename.endswith('.xlsx'):
                self.stock_data = pd.read_excel(file_path, index_col=0, parse_dates=True)
            else:
                print("지원하지 않는 파일 형식입니다.")
                return False
            
            # 데이터 검증
            if self.stock_data.empty:
                print("데이터가 비어있습니다.")
                return False
            
            # 결측치 처리
            self.stock_data = self.stock_data.fillna(method='ffill').dropna()
            
            # 기본 정보 설정
            self.n_assets = len(self.stock_data.columns)
            self.asset_names = list(self.stock_data.columns)
            
            print(f"✓ 데이터 로드 완료: {filename}")
            print(f"  - 기간: {self.stock_data.index[0].date()} ~ {self.stock_data.index[-1].date()}")
            print(f"  - 종목: {self.asset_names}")
            print(f"  - 데이터 크기: {self.stock_data.shape}")
            
            return True
            
        except Exception as e:
            print(f"데이터 로드 실패: {e}")
            return False
    
    def calculate_returns_stats(self):
        """수익률 통계 계산"""
        if self.stock_data is None:
            print("먼저 데이터를 로드하세요.")
            return False
        
        # 일간 수익률 계산
        self.returns = self.stock_data.pct_change().dropna()
        
        # 연간 기대수익률 (252 거래일)
        self.mu = self.returns.mean() * 252
        
        # 연간 공분산 행렬
        self.sigma = self.returns.cov() * 252
        
        print("✓ 수익률 통계 계산 완료")
        print(f"기대수익률 (연간):")
        for i, asset in enumerate(self.asset_names):
            print(f"  {asset}: {self.mu.iloc[i]:.2%}")
        
        return True
    
    def portfolio_stats(self, weights):
        """포트폴리오 통계 계산"""
        weights = np.array(weights)
        
        # 포트폴리오 기대수익률
        portfolio_return = np.sum(weights * self.mu)
        
        # 포트폴리오 분산
        portfolio_variance = np.dot(weights, np.dot(self.sigma, weights))
        portfolio_std = np.sqrt(portfolio_variance)
        
        # 샤프 비율 (무위험수익률 = 0 가정)
        sharpe_ratio = portfolio_return / portfolio_std if portfolio_std > 0 else 0
        
        return {
            'return': portfolio_return,
            'std': portfolio_std,
            'variance': portfolio_variance,
            'sharpe': sharpe_ratio
        }
    
    def min_variance_portfolio(self):
        """최소분산 포트폴리오"""
        # 제약조건: 가중치 합 = 1
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # 경계조건: 0 <= wi <= 1 (공매도 금지)
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # 초기값
        x0 = np.array([1/self.n_assets] * self.n_assets)
        
        # 목적함수: 분산 최소화
        def objective(weights):
            return self.portfolio_stats(weights)['variance']
        
        # 최적화 실행
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            stats = self.portfolio_stats(result.x)
            return {
                'weights': result.x,
                'return': stats['return'],
                'std': stats['std'],
                'sharpe': stats['sharpe'],
                'type': '최소분산'
            }
        else:
            print("최소분산 포트폴리오 최적화 실패")
            return None
    
    def max_sharpe_portfolio(self):
        """최대 샤프비율 포트폴리오"""
        # 제약조건: 가중치 합 = 1
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # 경계조건: 0 <= wi <= 1
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # 초기값
        x0 = np.array([1/self.n_assets] * self.n_assets)
        
        # 목적함수: 음의 샤프비율 (최소화 -> 최대화)
        def objective(weights):
            stats = self.portfolio_stats(weights)
            return -stats['sharpe'] if stats['sharpe'] > 0 else 1e6
        
        # 최적화 실행
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            stats = self.portfolio_stats(result.x)
            return {
                'weights': result.x,
                'return': stats['return'],
                'std': stats['std'],
                'sharpe': stats['sharpe'],
                'type': '최대샤프'
            }
        else:
            print("최대 샤프비율 포트폴리오 최적화 실패")
            return None
    
    def efficient_frontier(self, num_portfolios=100):
        """효율적 경계선 계산"""
        # 최소분산과 최대수익률 포트폴리오 구하기
        min_var = self.min_variance_portfolio()
        
        # 최대 기대수익률
        max_return = self.mu.max()
        min_return = min_var['return']
        
        # 목표 수익률 범위
        target_returns = np.linspace(min_return, max_return, num_portfolios)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            # 제약조건: 가중치 합 = 1, 목표 수익률
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x, target=target_return: np.sum(x * self.mu) - target}
            ]
            
            # 경계조건
            bounds = tuple((0, 1) for _ in range(self.n_assets))
            
            # 초기값
            x0 = np.array([1/self.n_assets] * self.n_assets)
            
            # 목적함수: 분산 최소화
            def objective(weights):
                return self.portfolio_stats(weights)['variance']
            
            # 최적화 실행
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                stats = self.portfolio_stats(result.x)
                efficient_portfolios.append({
                    'weights': result.x,
                    'return': stats['return'],
                    'std': stats['std'],
                    'sharpe': stats['sharpe']
                })
        
        return efficient_portfolios
    
    def monte_carlo_simulation(self, num_simulations=10000):
        """몬테카를로 시뮬레이션"""
        np.random.seed(42)  # 재현 가능한 결과
        
        results = []
        
        for _ in range(num_simulations):
            # 랜덤 가중치 생성
            weights = np.random.random(self.n_assets)
            weights = weights / np.sum(weights)  # 정규화
            
            # 포트폴리오 통계 계산
            stats = self.portfolio_stats(weights)
            results.append({
                'weights': weights,
                'return': stats['return'],
                'std': stats['std'],
                'sharpe': stats['sharpe']
            })
        
        return results
    
    def optimize_with_constraints(self, constraints_dict=None):
        """제약조건이 있는 최적화"""
        if constraints_dict is None:
            constraints_dict = {
                'max_weight': 0.4,  # 개별 종목 최대 40%
                'min_weight': 0.05,  # 개별 종목 최소 5%
                'max_sector_weight': None  # 섹터별 제한 (구현 시)
            }
        
        # 제약조건 설정
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # 경계조건
        bounds = []
        for i in range(self.n_assets):
            min_w = constraints_dict.get('min_weight', 0)
            max_w = constraints_dict.get('max_weight', 1)
            bounds.append((min_w, max_w))
        
        # 초기값
        x0 = np.array([1/self.n_assets] * self.n_assets)
        
        # 목적함수: 음의 샤프비율
        def objective(weights):
            stats = self.portfolio_stats(weights)
            return -stats['sharpe'] if stats['sharpe'] > 0 else 1e6
        
        # 최적화 실행
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            stats = self.portfolio_stats(result.x)
            return {
                'weights': result.x,
                'return': stats['return'],
                'std': stats['std'],
                'sharpe': stats['sharpe'],
                'type': '제약조건부최적'
            }
        else:
            print("제약조건부 최적화 실패")
            return None
    
    def risk_parity_portfolio(self):
        """위험균등 포트폴리오 (Risk Parity)"""
        # 각 자산의 위험 기여도가 동일하도록 하는 포트폴리오
        
        def risk_contribution(weights):
            """각 자산의 위험 기여도 계산"""
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.sigma, weights)))
            marginal_contrib = np.dot(self.sigma, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            return contrib
        
        def objective(weights):
            """위험 기여도의 분산 최소화"""
            contrib = risk_contribution(weights)
            target_contrib = np.ones(self.n_assets) / self.n_assets
            return np.sum((contrib - target_contrib * np.sum(contrib))**2)
        
        # 제약조건
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0.01, 0.99) for _ in range(self.n_assets))
        x0 = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            stats = self.portfolio_stats(result.x)
            return {
                'weights': result.x,
                'return': stats['return'],
                'std': stats['std'],
                'sharpe': stats['sharpe'],
                'type': '위험균등'
            }
        else:
            print("위험균등 포트폴리오 최적화 실패")
            return None
    
    def black_litterman_optimization(self, views=None, tau=0.025):
        """블랙-리터만 모델 (간단 버전)"""
        # 시장 균형 포트폴리오 (시가총액 가중 가정)
        market_caps = np.ones(self.n_assets)  # 동일 가정 (실제로는 시가총액 사용)
        w_market = market_caps / np.sum(market_caps)
        
        # 시장 균형 기대수익률 역산
        risk_aversion = 3  # 위험회피계수 가정
        pi = risk_aversion * np.dot(self.sigma, w_market)
        
        if views is None:
            # 뷰가 없으면 시장 균형 수익률 사용
            mu_bl = pi
        else:
            # 블랙-리터만 공식 적용 (간단 버전)
            # 실제로는 P, Q, Ω 행렬 필요
            mu_bl = pi  # 단순화
        
        # 블랙-리터만 공분산 행렬
        sigma_bl = self.sigma  # 단순화
        
        # 최적 포트폴리오 계산
        inv_sigma = np.linalg.inv(sigma_bl)
        ones = np.ones(self.n_assets)
        
        w_bl = np.dot(inv_sigma, mu_bl) / np.dot(ones, np.dot(inv_sigma, mu_bl))
        
        stats = self.portfolio_stats(w_bl)
        return {
            'weights': w_bl,
            'return': stats['return'],
            'std': stats['std'],
            'sharpe': stats['sharpe'],
            'type': '블랙리터만'
        }
    
    def analyze_portfolio(self, weights, portfolio_name="사용자 포트폴리오"):
        """포트폴리오 분석"""
        weights = np.array(weights)
        
        if abs(np.sum(weights) - 1.0) > 1e-6:
            print("경고: 가중치 합이 1이 아닙니다. 정규화합니다.")
            weights = weights / np.sum(weights)
        
        stats = self.portfolio_stats(weights)
        
        print(f"\n=== {portfolio_name} 분석 결과 ===")
        print(f"연간 기대수익률: {stats['return']:.2%}")
        print(f"연간 변동성:     {stats['std']:.2%}")
        print(f"샤프 비율:       {stats['sharpe']:.4f}")
        
        print(f"\n자산 배분:")
        for i, asset in enumerate(self.asset_names):
            print(f"  {asset}: {weights[i]:.1%}")
        
        # 위험 기여도 분석
        portfolio_vol = stats['std']
        marginal_contrib = np.dot(self.sigma, weights) / portfolio_vol
        risk_contrib = weights * marginal_contrib
        
        print(f"\n위험 기여도:")
        for i, asset in enumerate(self.asset_names):
            print(f"  {asset}: {risk_contrib[i]/np.sum(risk_contrib):.1%}")
        
        return stats
    
    def plot_efficient_frontier(self, efficient_portfolios, optimal_portfolios=None):
        """효율적 경계선 시각화"""
        if not efficient_portfolios:
            print("효율적 경계선 데이터가 없습니다.")
            return
        
        # 데이터 추출
        ef_returns = [p['return'] for p in efficient_portfolios]
        ef_stds = [p['std'] for p in efficient_portfolios]
        
        plt.figure(figsize=(12, 8))
        
        # 효율적 경계선
        plt.plot(ef_stds, ef_returns, 'b-', linewidth=2, label='효율적 경계선')
        
        # 개별 자산
        individual_returns = self.mu.values
        individual_stds = np.sqrt(np.diag(self.sigma))
        
        plt.scatter(individual_stds, individual_returns, 
                   c='red', s=100, alpha=0.7, label='개별 자산')
        
        # 개별 자산 라벨
        for i, asset in enumerate(self.asset_names):
            plt.annotate(asset, (individual_stds[i], individual_returns[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # 최적 포트폴리오들
        if optimal_portfolios:
            colors = ['green', 'orange', 'purple', 'brown', 'pink']
            for i, portfolio in enumerate(optimal_portfolios):
                if portfolio:
                    plt.scatter(portfolio['std'], portfolio['return'], 
                              c=colors[i % len(colors)], s=150, 
                              label=portfolio['type'], marker='*')
        
        plt.xlabel('위험 (표준편차)')
        plt.ylabel('기대수익률')
        plt.title('효율적 경계선과 최적 포트폴리오')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 저장
        save_path = os.path.join(self.data_dir, 'efficient_frontier.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"효율적 경계선 차트 저장: {save_path}")
        plt.show()
    
    def plot_monte_carlo(self, mc_results):
        """몬테카를로 시뮬레이션 결과 시각화"""
        if not mc_results:
            print("몬테카를로 데이터가 없습니다.")
            return
        
        returns = [r['return'] for r in mc_results]
        stds = [r['std'] for r in mc_results]
        sharpes = [r['sharpe'] for r in mc_results]
        
        plt.figure(figsize=(15, 5))
        
        # 위험-수익률 산점도
        plt.subplot(1, 3, 1)
        scatter = plt.scatter(stds, returns, c=sharpes, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='샤프 비율')
        plt.xlabel('위험 (표준편차)')
        plt.ylabel('기대수익률')
        plt.title('몬테카를로 시뮬레이션')
        plt.grid(True, alpha=0.3)
        
        # 수익률 분포
        plt.subplot(1, 3, 2)
        plt.hist(returns, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('기대수익률')
        plt.ylabel('빈도')
        plt.title('수익률 분포')
        plt.grid(True, alpha=0.3)
        
        # 샤프비율 분포
        plt.subplot(1, 3, 3)
        plt.hist(sharpes, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('샤프 비율')
        plt.ylabel('빈도')
        plt.title('샤프 비율 분포')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 저장
        save_path = os.path.join(self.data_dir, 'monte_carlo_simulation.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"몬테카를로 차트 저장: {save_path}")
        plt.show()
    
    def save_results(self, results, filename=None):
        """결과를 Excel로 저장"""
        if filename is None:
            filename = f"portfolio_optimization_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        
        file_path = os.path.join(self.data_dir, filename)
        
        try:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # 최적 포트폴리오들
                portfolio_summary = []
                for result in results:
                    if result:
                        row = {
                            '포트폴리오유형': result['type'],
                            '기대수익률': result['return'],
                            '표준편차': result['std'],
                            '샤프비율': result['sharpe']
                        }
                        # 가중치 추가
                        for i, asset in enumerate(self.asset_names):
                            row[f'{asset}_비중'] = result['weights'][i]
                        portfolio_summary.append(row)
                
                summary_df = pd.DataFrame(portfolio_summary)
                summary_df.to_excel(writer, sheet_name='최적포트폴리오', index=False)
                
                # 기대수익률과 공분산 행렬
                mu_df = pd.DataFrame({'자산': self.asset_names, '기대수익률': self.mu.values})
                mu_df.to_excel(writer, sheet_name='기대수익률', index=False)
                
                self.sigma.to_excel(writer, sheet_name='공분산행렬')
                
                # 상관관계 행렬
                corr_matrix = self.returns.corr()
                corr_matrix.to_excel(writer, sheet_name='상관관계행렬')
                
            print(f"✓ 결과 저장 완료: {filename}")
            return file_path
            
        except Exception as e:
            print(f"결과 저장 실패: {e}")
            return None

def main():
    """메인 실행 함수"""
    print("="*60)
    print("         포트폴리오 최적화 엔진")
    print("="*60)
    
    # 옵티마이저 초기화
    optimizer = PortfolioOptimizer()
    
    # 데이터 로드
    if not optimizer.load_data():
        return
    
    # 수익률 통계 계산
    if not optimizer.calculate_returns_stats():
        return
    
    # 최적화 실행
    print("\n포트폴리오 최적화를 실행합니다...")
    
    optimal_portfolios = []
    
    # 1. 최소분산 포트폴리오
    print("\n1. 최소분산 포트폴리오 계산...")
    min_var = optimizer.min_variance_portfolio()
    if min_var:
        optimal_portfolios.append(min_var)
        optimizer.analyze_portfolio(min_var['weights'], min_var['type'])
    
    # 2. 최대 샤프비율 포트폴리오
    print("\n2. 최대 샤프비율 포트폴리오 계산...")
    max_sharpe = optimizer.max_sharpe_portfolio()
    if max_sharpe:
        optimal_portfolios.append(max_sharpe)
        optimizer.analyze_portfolio(max_sharpe['weights'], max_sharpe['type'])
    
    # 3. 제약조건부 최적화
    print("\n3. 제약조건부 최적화 계산...")
    constrained = optimizer.optimize_with_constraints({
        'max_weight': 0.4,
        'min_weight': 0.05
    })
    if constrained:
        optimal_portfolios.append(constrained)
        optimizer.analyze_portfolio(constrained['weights'], constrained['type'])
    
    # 4. 위험균등 포트폴리오
    print("\n4. 위험균등 포트폴리오 계산...")
    risk_parity = optimizer.risk_parity_portfolio()
    if risk_parity:
        optimal_portfolios.append(risk_parity)
        optimizer.analyze_portfolio(risk_parity['weights'], risk_parity['type'])
    
    # 5. 균등가중 포트폴리오 (비교용)
    equal_weights = np.array([1/optimizer.n_assets] * optimizer.n_assets)
    equal_portfolio = {
        'weights': equal_weights,
        'type': '균등가중'
    }
    equal_portfolio.update(optimizer.portfolio_stats(equal_weights))
    optimal_portfolios.append(equal_portfolio)
    optimizer.analyze_portfolio(equal_weights, "균등가중")
    
    # 효율적 경계선 계산
    print("\n효율적 경계선 계산 중...")
    efficient_portfolios = optimizer.efficient_frontier(50)
    
    # 몬테카를로 시뮬레이션
    print("몬테카를로 시뮬레이션 실행 중...")
    mc_results = optimizer.monte_carlo_simulation(5000)
    
    # 시각화
    print("\n시각화 생성 중...")
    optimizer.plot_efficient_frontier(efficient_portfolios, optimal_portfolios)
    optimizer.plot_monte_carlo(mc_results)
    
    # 결과 저장
    print("\n결과 저장 중...")
    optimizer.save_results(optimal_portfolios)
    
    # 최종 요약
    print("\n" + "="*60)
    print("         최적화 결과 요약")
    print("="*60)
    
    # 샤프비율 기준 순위
    valid_portfolios = [p for p in optimal_portfolios if p and p['sharpe'] > 0]
    sorted_portfolios = sorted(valid_portfolios, key=lambda x: x['sharpe'], reverse=True)
    
    print("샤프비율 기준 포트폴리오 순위:")
    for i, portfolio in enumerate(sorted_portfolios, 1):
        print(f"{i}. {portfolio['type']}: 샤프비율 {portfolio['sharpe']:.4f}, "
              f"수익률 {portfolio['return']:.2%}, 위험 {portfolio['std']:.2%}")
    
    # 1000만원 투자 시뮬레이션
    print(f"\n💰 1000만원 투자 시뮬레이션 (3년 후 예상 금액):")
    initial_investment = 10_000_000
    years = 3
    
    for portfolio in sorted_portfolios[:5]:  # 상위 5개만
        annual_return = portfolio['return']
        final_amount = initial_investment * (1 + annual_return) ** years
        profit = final_amount - initial_investment
        
        print(f"  {portfolio['type']:12s}: {final_amount:>12,.0f}원 "
              f"(+{profit:>8,.0f}원, {annual_return:.1%}/년)")
    
    print(f"\n🎯 추천 포트폴리오: {sorted_portfolios[0]['type']}")
    print("="*60)

def interactive_mode():
    """대화형 모드"""
    optimizer = PortfolioOptimizer()
    
    while True:
        print("\n" + "="*50)
        print("        포트폴리오 최적화 도구")
        print("="*50)
        print("1. 데이터 로드")
        print("2. 전체 최적화 실행")
        print("3. 개별 최적화 선택")
        print("4. 사용자 포트폴리오 분석")
        print("5. 효율적 경계선 계산")
        print("6. 몬테카를로 시뮬레이션")
        print("7. 제약조건 설정")
        print("8. 결과 내보내기")
        print("9. 종료")
        print("="*50)
        
        choice = input("선택하세요 (1-9): ").strip()
        
        if choice == '1':
            optimizer.load_data()
            if optimizer.stock_data is not None:
                optimizer.calculate_returns_stats()
                
        elif choice == '2':
            if optimizer.stock_data is None:
                print("먼저 데이터를 로드하세요.")
                continue
            main_with_optimizer(optimizer)
            
        elif choice == '3':
            if optimizer.stock_data is None:
                print("먼저 데이터를 로드하세요.")
                continue
            individual_optimization_menu(optimizer)
            
        elif choice == '4':
            if optimizer.stock_data is None:
                print("먼저 데이터를 로드하세요.")
                continue
            analyze_user_portfolio(optimizer)
            
        elif choice == '5':
            if optimizer.stock_data is None:
                print("먼저 데이터를 로드하세요.")
                continue
            print("효율적 경계선 계산 중...")
            efficient_portfolios = optimizer.efficient_frontier(100)
            optimizer.plot_efficient_frontier(efficient_portfolios)
            
        elif choice == '6':
            if optimizer.stock_data is None:
                print("먼저 데이터를 로드하세요.")
                continue
            num_sims = input("시뮬레이션 횟수 (기본값: 10000): ").strip()
            try:
                num_sims = int(num_sims) if num_sims else 10000
            except ValueError:
                num_sims = 10000
            
            print(f"몬테카를로 시뮬레이션 실행 중... ({num_sims:,}회)")
            mc_results = optimizer.monte_carlo_simulation(num_sims)
            optimizer.plot_monte_carlo(mc_results)
            
        elif choice == '7':
            if optimizer.stock_data is None:
                print("먼저 데이터를 로드하세요.")
                continue
            constraint_menu(optimizer)
            
        elif choice == '8':
            if optimizer.stock_data is None:
                print("먼저 데이터를 로드하세요.")
                continue
            export_menu(optimizer)
            
        elif choice == '9':
            print("프로그램을 종료합니다.")
            break
            
        else:
            print("잘못된 선택입니다. 1-9 중에서 선택해주세요.")

def main_with_optimizer(optimizer):
    """기존 optimizer로 전체 최적화 실행"""
    optimal_portfolios = []
    
    print("\n전체 최적화를 실행합니다...")
    
    # 모든 최적화 방법 실행
    methods = [
        ("최소분산", optimizer.min_variance_portfolio),
        ("최대샤프", optimizer.max_sharpe_portfolio),
        ("제약조건부", lambda: optimizer.optimize_with_constraints()),
        ("위험균등", optimizer.risk_parity_portfolio),
    ]
    
    for method_name, method_func in methods:
        print(f"\n{method_name} 포트폴리오 계산 중...")
        try:
            result = method_func()
            if result:
                optimal_portfolios.append(result)
                optimizer.analyze_portfolio(result['weights'], result['type'])
        except Exception as e:
            print(f"{method_name} 최적화 실패: {e}")
    
    # 균등가중 추가
    equal_weights = np.array([1/optimizer.n_assets] * optimizer.n_assets)
    equal_portfolio = {'weights': equal_weights, 'type': '균등가중'}
    equal_portfolio.update(optimizer.portfolio_stats(equal_weights))
    optimal_portfolios.append(equal_portfolio)
    
    # 효율적 경계선과 시각화
    print("\n효율적 경계선 계산 및 시각화...")
    efficient_portfolios = optimizer.efficient_frontier(50)
    optimizer.plot_efficient_frontier(efficient_portfolios, optimal_portfolios)
    
    # 결과 저장
    optimizer.save_results(optimal_portfolios)
    
    return optimal_portfolios

def individual_optimization_menu(optimizer):
    """개별 최적화 메뉴"""
    while True:
        print("\n" + "="*40)
        print("개별 최적화 선택")
        print("="*40)
        print("1. 최소분산 포트폴리오")
        print("2. 최대 샤프비율 포트폴리오")
        print("3. 제약조건부 최적화")
        print("4. 위험균등 포트폴리오")
        print("5. 블랙-리터만 모델")
        print("6. 돌아가기")
        print("="*40)
        
        choice = input("선택하세요 (1-6): ").strip()
        
        if choice == '1':
            result = optimizer.min_variance_portfolio()
            if result:
                optimizer.analyze_portfolio(result['weights'], result['type'])
        elif choice == '2':
            result = optimizer.max_sharpe_portfolio()
            if result:
                optimizer.analyze_portfolio(result['weights'], result['type'])
        elif choice == '3':
            result = optimizer.optimize_with_constraints()
            if result:
                optimizer.analyze_portfolio(result['weights'], result['type'])
        elif choice == '4':
            result = optimizer.risk_parity_portfolio()
            if result:
                optimizer.analyze_portfolio(result['weights'], result['type'])
        elif choice == '5':
            result = optimizer.black_litterman_optimization()
            if result:
                optimizer.analyze_portfolio(result['weights'], result['type'])
        elif choice == '6':
            break
        else:
            print("잘못된 선택입니다.")

def analyze_user_portfolio(optimizer):
    """사용자 포트폴리오 분석"""
    print("\n사용자 포트폴리오 입력")
    print(f"종목: {optimizer.asset_names}")
    
    weights = []
    total_weight = 0
    
    for asset in optimizer.asset_names:
        while True:
            try:
                weight_input = input(f"{asset} 비중 (%, 0-100): ").strip()
                weight = float(weight_input) / 100
                if 0 <= weight <= 1:
                    weights.append(weight)
                    total_weight += weight
                    break
                else:
                    print("0-100 사이의 값을 입력하세요.")
            except ValueError:
                print("숫자를 입력하세요.")
    
    if abs(total_weight - 1.0) > 1e-6:
        print(f"가중치 합: {total_weight:.1%}")
        normalize = input("자동으로 정규화하시겠습니까? (y/N): ").strip().lower()
        if normalize == 'y':
            weights = [w/total_weight for w in weights]
        else:
            print("가중치 합이 100%가 되도록 다시 입력해주세요.")
            return
    
    # 분석 실행
    optimizer.analyze_portfolio(weights, "사용자 포트폴리오")
    
    # 다른 포트폴리오와 비교
    compare = input("\n다른 포트폴리오와 비교하시겠습니까? (y/N): ").strip().lower()
    if compare == 'y':
        compare_portfolios(optimizer, weights)

def compare_portfolios(optimizer, user_weights):
    """포트폴리오 비교"""
    portfolios = []
    
    # 사용자 포트폴리오
    user_stats = optimizer.portfolio_stats(user_weights)
    portfolios.append({
        'name': '사용자 포트폴리오',
        'weights': user_weights,
        **user_stats
    })
    
    # 최적 포트폴리오들
    methods = [
        ("최소분산", optimizer.min_variance_portfolio),
        ("최대샤프", optimizer.max_sharpe_portfolio),
        ("균등가중", lambda: {'weights': [1/optimizer.n_assets]*optimizer.n_assets, 'type': '균등가중'})
    ]
    
    for name, method in methods:
        try:
            result = method()
            if result:
                if 'return' not in result:
                    result.update(optimizer.portfolio_stats(result['weights']))
                portfolios.append({
                    'name': name,
                    'weights': result['weights'],
                    'return': result['return'],
                    'std': result['std'],
                    'sharpe': result['sharpe']
                })
        except Exception as e:
            print(f"{name} 계산 실패: {e}")
    
    # 비교 테이블 출력
    print("\n" + "="*80)
    print("포트폴리오 비교")
    print("="*80)
    print(f"{'포트폴리오':<15} {'수익률':<8} {'위험':<8} {'샤프비율':<10} {'1000만원→3년후':<15}")
    print("-"*80)
    
    for portfolio in portfolios:
        final_amount = 10_000_000 * (1 + portfolio['return']) ** 3
        print(f"{portfolio['name']:<15} {portfolio['return']:7.2%} {portfolio['std']:7.2%} "
              f"{portfolio['sharpe']:9.4f} {final_amount:>12,.0f}원")

def constraint_menu(optimizer):
    """제약조건 설정 메뉴"""
    print("\n제약조건 설정")
    
    constraints = {}
    
    # 개별 종목 최대 비중
    max_weight = input("개별 종목 최대 비중 (%, 기본값: 40): ").strip()
    try:
        constraints['max_weight'] = float(max_weight) / 100 if max_weight else 0.4
    except ValueError:
        constraints['max_weight'] = 0.4
    
    # 개별 종목 최소 비중
    min_weight = input("개별 종목 최소 비중 (%, 기본값: 5): ").strip()
    try:
        constraints['min_weight'] = float(min_weight) / 100 if min_weight else 0.05
    except ValueError:
        constraints['min_weight'] = 0.05
    
    print(f"\n설정된 제약조건:")
    print(f"  개별 종목 최대 비중: {constraints['max_weight']:.1%}")
    print(f"  개별 종목 최소 비중: {constraints['min_weight']:.1%}")
    
    # 최적화 실행
    result = optimizer.optimize_with_constraints(constraints)
    if result:
        optimizer.analyze_portfolio(result['weights'], "제약조건부 최적")

def export_menu(optimizer):
    """결과 내보내기 메뉴"""
    print("\n결과 내보내기")
    print("1. 모든 최적 포트폴리오 계산 후 저장")
    print("2. 현재 결과만 저장")
    print("3. 효율적 경계선 데이터 저장")
    
    choice = input("선택하세요 (1-3): ").strip()
    
    if choice == '1':
        portfolios = main_with_optimizer(optimizer)
        optimizer.save_results(portfolios)
    elif choice == '2':
        # 간단한 결과 저장
        equal_weights = np.array([1/optimizer.n_assets] * optimizer.n_assets)
        equal_portfolio = {'weights': equal_weights, 'type': '균등가중'}
        equal_portfolio.update(optimizer.portfolio_stats(equal_weights))
        optimizer.save_results([equal_portfolio])
    elif choice == '3':
        print("효율적 경계선 계산 중...")
        efficient_portfolios = optimizer.efficient_frontier(100)
        
        # 효율적 경계선 데이터를 DataFrame으로 변환
        ef_data = []
        for i, portfolio in enumerate(efficient_portfolios):
            row = {
                '포트폴리오번호': i+1,
                '기대수익률': portfolio['return'],
                '표준편차': portfolio['std'],
                '샤프비율': portfolio['sharpe']
            }
            for j, asset in enumerate(optimizer.asset_names):
                row[f'{asset}_비중'] = portfolio['weights'][j]
            ef_data.append(row)
        
        ef_df = pd.DataFrame(ef_data)
        filename = f"efficient_frontier_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        file_path = os.path.join(optimizer.data_dir, filename)
        ef_df.to_excel(file_path, index=False)
        print(f"효율적 경계선 데이터 저장: {filename}")

# 실행 부분
if __name__ == "__main__":
    print("포트폴리오 최적화 엔진을 시작합니다.")
    print("1. 자동 실행")
    print("2. 대화형 모드")
    
    mode = input("모드를 선택하세요 (1-2): ").strip()
    
    if mode == '1':
        main()
    else:
        interactive_mode()

# 추가 유틸리티 함수들

def calculate_var_cvar(returns, confidence_level=0.05):
    """VaR과 CVaR 계산"""
    var = np.percentile(returns, confidence_level * 100)
    cvar = returns[returns <= var].mean()
    return var, cvar

def calculate_maximum_drawdown(cumulative_returns):
    """최대 낙폭 계산"""
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdown)
    return max_drawdown

def backtest_portfolio(optimizer, weights, rebalance_freq='monthly'):
    """포트폴리오 백테스트 (간단 버전)"""
    if optimizer.stock_data is None:
        print("데이터가 로드되지 않았습니다.")
        return None
    
    prices = optimizer.stock_data
    weights = np.array(weights)
    
    # 수익률 계산
    returns = prices.pct_change().dropna()
    
    # 포트폴리오 수익률 시계열
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # 누적 수익률
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    # 성과 지표 계산
    total_return = cumulative_returns.iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
    annual_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility
    max_drawdown = calculate_maximum_drawdown(cumulative_returns)
    
    var_5, cvar_5 = calculate_var_cvar(portfolio_returns)
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'var_5': var_5,
        'cvar_5': cvar_5,
        'cumulative_returns': cumulative_returns,
        'portfolio_returns': portfolio_returns
    }

# 사용 예시 및 가이드
"""
=== 포트폴리오 최적화 엔진 사용 가이드 ===

📦 필요한 라이브러리:
pip install pandas numpy matplotlib seaborn scipy openpyxl

🚀 실행 방법:
1. 자동 실행: python portfolio_optimizer.py
2. 대화형: 실행 후 2번 선택

📁 지원 파일 형식:
- pkl: 캐시된 주가 데이터
- csv: CSV 형식 주가 데이터  
- xlsx: Excel 형식 주가 데이터

🎯 제공 기능:
1. 최소분산 포트폴리오
2. 최대 샤프비율 포트폴리오
3. 제약조건부 최적화
4. 위험균등 포트폴리오
5. 블랙-리터만 모델
6. 효율적 경계선 계산
7. 몬테카를로 시뮬레이션
8. 사용자 포트폴리오 분석

📊 수학적 모델:
- 현대 포트폴리오 이론 (MPT)
- 라그랑주 승수법
- 이차계획법 (Quadratic Programming)
- 몬테카를로 시뮬레이션
- 베이지안 최적화

📈 결과물:
- Excel 형식 상세 분석 결과
- 효율적 경계선 차트
- 몬테카를로 시뮬레이션 차트
- 포트폴리오 비교 분석

💡 사용 팁:
1. data 폴더에 주가 데이터 파일 준비
2. 대화형 모드로 단계별 분석
3. 제약조건 설정으로 실무적 포트폴리오 구성
4. 백테스트로 과거 성과 검증

🔧 고급 기능:
- 사용자 정의 제약조건
- 리스크 패리티 모델
- VaR/CVaR 리스크 측정
- 최대 낙폭 분석
- 다양한 리밸런싱 전략
"""
