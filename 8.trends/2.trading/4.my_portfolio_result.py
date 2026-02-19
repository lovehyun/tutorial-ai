# 포트폴리오 최적화 엔진
# 효율적 경계선, 샤프비율 최적화, 몬테카를로 시뮬레이션 + 사용자 조합 하이라이트(빨간 X)

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

# =========================
# 추가 유틸리티 함수들 (백테스트 등)
# =========================
def calculate_var_cvar(returns, confidence_level=0.05):
    """VaR과 CVaR 계산 (수익률 시리즈 입력, 하위 5% 손실 평균 등)"""
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
    """포트폴리오 백테스트 (간단 버전: buy&hold 근사)"""
    if optimizer.stock_data is None:
        print("데이터가 로드되지 않았습니다.")
        return None

    prices = optimizer.stock_data
    weights = np.array(weights)

    # 일간 수익률
    returns = prices.pct_change().dropna()

    # 포트폴리오 수익률 시계열 (고정 가중치 가정)
    portfolio_returns = (returns * weights).sum(axis=1)

    # 누적 수익률
    cumulative_returns = (1 + portfolio_returns).cumprod()

    # 성과 지표 계산
    total_return = cumulative_returns.iloc[-1] - 1
    n_days = len(portfolio_returns)
    if n_days == 0:
        return None
    annual_return = (1 + total_return) ** (252 / n_days) - 1
    annual_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return / annual_volatility) if annual_volatility > 0 else np.nan
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

# =========================
# 최적화 클래스
# =========================
class PortfolioOptimizer:
    """포트폴리오 최적화 클래스"""

    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.stock_data = None
        self.returns = None
        self.mu = None      # 기대수익률 벡터(연)
        self.sigma = None   # 공분산 행렬(연)
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

            # 파일 형식별 로드
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

            # 데이터 검증/결측치 처리
            if self.stock_data.empty:
                print("데이터가 비어있습니다.")
                return False

            self.stock_data = self.stock_data.fillna(method='ffill').dropna()

            # 기본 정보
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

        # 일간 수익률
        self.returns = self.stock_data.pct_change().dropna()

        # 연간 기대수익률/공분산
        self.mu = self.returns.mean() * 252
        self.sigma = self.returns.cov() * 252

        print("✓ 수익률 통계 계산 완료")
        print("기대수익률 (연간):")
        for i, asset in enumerate(self.asset_names):
            print(f"  {asset}: {self.mu.iloc[i]:.2%}")

        return True

    def portfolio_stats(self, weights):
        """포트폴리오 통계 계산"""
        weights = np.array(weights)
        portfolio_return = np.sum(weights * self.mu)
        portfolio_variance = np.dot(weights, np.dot(self.sigma, weights))
        portfolio_std = np.sqrt(portfolio_variance)
        sharpe_ratio = portfolio_return / portfolio_std if portfolio_std > 0 else 0.0
        return {
            'return': portfolio_return,
            'std': portfolio_std,
            'variance': portfolio_variance,
            'sharpe': sharpe_ratio
        }

    def min_variance_portfolio(self):
        """최소분산 포트폴리오"""
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        x0 = np.array([1/self.n_assets] * self.n_assets)

        def objective(weights):
            return self.portfolio_stats(weights)['variance']

        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            stats = self.portfolio_stats(result.x)
            return {'weights': result.x, 'return': stats['return'], 'std': stats['std'],
                    'sharpe': stats['sharpe'], 'type': '최소분산'}
        else:
            print("최소분산 포트폴리오 최적화 실패")
            return None

    def max_sharpe_portfolio(self):
        """최대 샤프비율 포트폴리오"""
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        x0 = np.array([1/self.n_assets] * self.n_assets)

        def objective(weights):
            stats = self.portfolio_stats(weights)
            return -stats['sharpe'] if stats['sharpe'] > 0 else 1e6

        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            stats = self.portfolio_stats(result.x)
            return {'weights': result.x, 'return': stats['return'], 'std': stats['std'],
                    'sharpe': stats['sharpe'], 'type': '최대샤프'}
        else:
            print("최대 샤프비율 포트폴리오 최적화 실패")
            return None

    def efficient_frontier(self, num_portfolios=100):
        """효율적 경계선 계산"""
        min_var = self.min_variance_portfolio()
        max_return = self.mu.max()
        min_return = min_var['return']

        target_returns = np.linspace(min_return, max_return, num_portfolios)
        efficient_portfolios = []

        for target_return in target_returns:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x, target=target_return: np.sum(x * self.mu) - target}
            ]
            bounds = tuple((0, 1) for _ in range(self.n_assets))
            x0 = np.array([1/self.n_assets] * self.n_assets)

            def objective(weights):
                return self.portfolio_stats(weights)['variance']

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
        np.random.seed(42)
        results = []
        for _ in range(num_simulations):
            weights = np.random.random(self.n_assets)
            weights = weights / np.sum(weights)
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
            constraints_dict = {'max_weight': 0.4, 'min_weight': 0.05}

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        bounds = []
        for _ in range(self.n_assets):
            min_w = constraints_dict.get('min_weight', 0)
            max_w = constraints_dict.get('max_weight', 1)
            bounds.append((min_w, max_w))

        x0 = np.array([1/self.n_assets] * self.n_assets)

        def objective(weights):
            stats = self.portfolio_stats(weights)
            return -stats['sharpe'] if stats['sharpe'] > 0 else 1e6

        result = minimize(objective, x0, method='SLSQP', bounds=tuple(bounds), constraints=constraints)

        if result.success:
            stats = self.portfolio_stats(result.x)
            return {'weights': result.x, 'return': stats['return'], 'std': stats['std'],
                    'sharpe': stats['sharpe'], 'type': '제약조건부최적'}
        else:
            print("제약조건부 최적화 실패")
            return None

    def risk_parity_portfolio(self):
        """위험균등 포트폴리오 (Risk Parity)"""
        def risk_contribution(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.sigma, weights)))
            marginal_contrib = np.dot(self.sigma, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            return contrib

        def objective(weights):
            contrib = risk_contribution(weights)
            target_contrib = np.ones(self.n_assets) / self.n_assets
            return np.sum((contrib - target_contrib * np.sum(contrib))**2)

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0.01, 0.99) for _ in range(self.n_assets))
        x0 = np.array([1/self.n_assets] * self.n_assets)

        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            stats = self.portfolio_stats(result.x)
            return {'weights': result.x, 'return': stats['return'], 'std': stats['std'],
                    'sharpe': stats['sharpe'], 'type': '위험균등'}
        else:
            print("위험균등 포트폴리오 최적화 실패")
            return None

    def black_litterman_optimization(self, views=None, tau=0.025):
        """블랙-리터만 모델 (간단 버전)"""
        market_caps = np.ones(self.n_assets)
        w_market = market_caps / np.sum(market_caps)
        risk_aversion = 3
        pi = risk_aversion * np.dot(self.sigma, w_market)
        mu_bl = pi  # 단순화
        sigma_bl = self.sigma
        inv_sigma = np.linalg.inv(sigma_bl)
        ones = np.ones(self.n_assets)
        w_bl = np.dot(inv_sigma, mu_bl) / np.dot(ones, np.dot(inv_sigma, mu_bl))
        stats = self.portfolio_stats(w_bl)
        return {'weights': w_bl, 'return': stats['return'], 'std': stats['std'],
                'sharpe': stats['sharpe'], 'type': '블랙리터만'}

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

        portfolio_vol = stats['std']
        if portfolio_vol > 0:
            marginal_contrib = np.dot(self.sigma, weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib
            print(f"\n위험 기여도:")
            for i, asset in enumerate(self.asset_names):
                print(f"  {asset}: {risk_contrib[i]/np.sum(risk_contrib):.1%}")

        return stats

    def plot_efficient_frontier(self, efficient_portfolios, optimal_portfolios=None, highlight=None):
        """효율적 경계선 시각화 (+ 사용자 조합 빨간 X 표시 옵션)"""
        if not efficient_portfolios:
            print("효율적 경계선 데이터가 없습니다.")
            return

        ef_returns = [p['return'] for p in efficient_portfolios]
        ef_stds = [p['std'] for p in efficient_portfolios]

        plt.figure(figsize=(12, 8))
        plt.plot(ef_stds, ef_returns, 'b-', linewidth=2, label='효율적 경계선')

        # 개별 자산
        individual_returns = self.mu.values
        individual_stds = np.sqrt(np.diag(self.sigma))
        plt.scatter(individual_stds, individual_returns, c='red', s=100, alpha=0.7, label='개별 자산')
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

        # 사용자 조합 하이라이트
        if highlight is not None:
            plt.scatter(highlight['std'], highlight['return'],
                        marker='x', s=220, c='red', linewidths=3,
                        label=highlight.get('label', '사용자 조합'))
            if 'period_total_return' in highlight and highlight['period_total_return'] == highlight['period_total_return']:
                txt = f"{highlight.get('label','사용자 조합')}\n기간수익률: {highlight['period_total_return']:.2%}"
                plt.annotate(txt, (highlight['std'], highlight['return']),
                             xytext=(10, 10), textcoords='offset points', fontsize=9)

        plt.xlabel('위험 (표준편차)')
        plt.ylabel('기대수익률')
        plt.title('효율적 경계선과 최적 포트폴리오')
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(self.data_dir, 'efficient_frontier.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"효율적 경계선 차트 저장: {save_path}")
        plt.show()

    def plot_monte_carlo(self, mc_results, highlight=None):
        """몬테카를로 시뮬레이션 결과 시각화 (+ 사용자 조합 빨간 X 표시 옵션)"""
        if not mc_results:
            print("몬테카를로 데이터가 없습니다.")
            return

        returns = [r['return'] for r in mc_results]
        stds = [r['std'] for r in mc_results]
        sharpes = [r['sharpe'] for r in mc_results]

        plt.figure(figsize=(15, 5))

        # (1) 위험-수익률 산점도
        plt.subplot(1, 3, 1)
        scatter = plt.scatter(stds, returns, c=sharpes, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='샤프 비율')
        plt.xlabel('위험 (표준편차)')
        plt.ylabel('기대수익률')
        plt.title('몬테카를로 시뮬레이션')
        plt.grid(True, alpha=0.3)

        if highlight is not None:
            plt.scatter(highlight['std'], highlight['return'],
                        marker='x', s=180, c='red', linewidths=3,
                        label=highlight.get('label', '사용자 조합'))
            plt.legend()

        # (2) 수익률 분포
        plt.subplot(1, 3, 2)
        plt.hist(returns, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('기대수익률')
        plt.ylabel('빈도')
        plt.title('수익률 분포')
        plt.grid(True, alpha=0.3)

        # (3) 샤프비율 분포
        plt.subplot(1, 3, 3)
        plt.hist(sharpes, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('샤프 비율')
        plt.ylabel('빈도')
        plt.title('샤프 비율 분포')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
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
                # 최적 포트폴리오 요약
                portfolio_summary = []
                for result in results:
                    if result:
                        row = {
                            '포트폴리오유형': result['type'],
                            '기대수익률': result['return'],
                            '표준편차': result['std'],
                            '샤프비율': result['sharpe']
                        }
                        for i, asset in enumerate(self.asset_names):
                            row[f'{asset}_비중'] = result['weights'][i]
                        portfolio_summary.append(row)

                summary_df = pd.DataFrame(portfolio_summary)
                summary_df.to_excel(writer, sheet_name='최적포트폴리오', index=False)

                # 기대수익률/공분산/상관
                mu_df = pd.DataFrame({'자산': self.asset_names, '기대수익률': self.mu.values})
                mu_df.to_excel(writer, sheet_name='기대수익률', index=False)

                self.sigma.to_excel(writer, sheet_name='공분산행렬')

                corr_matrix = self.returns.corr()
                corr_matrix.to_excel(writer, sheet_name='상관관계행렬')

            print(f"✓ 결과 저장 완료: {filename}")
            return file_path

        except Exception as e:
            print(f"결과 저장 실패: {e}")
            return None

    # ===== 사용자 조합 하이라이트용 추가 메서드들 =====
    def _build_weight_vector(self, selection, normalize=True, detect_percent=True):
        """
        selection: dict({'삼성전자': 50, 'KB금융':30, ...}) 또는
                   dict({'삼성전자':0.5, 'KB금융':0.3, ...}) 모두 허용
        normalize=True면 합이 1이 아니어도 자동 정규화
        """
        if not isinstance(selection, dict):
            raise ValueError("selection은 {'종목': 비중, ...} dict 형태로 주세요.")
        assets = list(selection.keys())
        weights_in = np.array(list(selection.values()), dtype=float)

        # 퍼센트(%)로 보이면 0~1로 변환
        if detect_percent and (np.any(weights_in > 1.0) or weights_in.sum() > 1.0):
            weights_in = weights_in / 100.0

        # 전체 자산 길이에 맞춰 풀 벡터 구성
        full = np.zeros(self.n_assets, dtype=float)
        for a, w in zip(assets, weights_in):
            if a not in self.asset_names:
                raise ValueError(f"[오류] 데이터에 없는 종목: {a}")
            idx = self.asset_names.index(a)
            full[idx] = w

        if normalize:
            s = full.sum()
            if s <= 0:
                raise ValueError("가중치 합이 0입니다.")
            full = full / s
        return full

    def user_portfolio_point(self, selection, label="사용자 조합"):
        """
        selection: {'종목명': 비중, ...} (비중은 % 또는 0~1 모두 허용)
        반환: 그래프 하이라이트와 기간 수익률 지표를 포함한 dict
        """
        if self.stock_data is None or self.mu is None or self.sigma is None:
            raise RuntimeError("먼저 데이터 로드 및 수익률 통계를 계산하세요.")

        weights = self._build_weight_vector(selection, normalize=True, detect_percent=True)
        stats = self.portfolio_stats(weights)

        # 기간 수익률(백테스트)
        perf = backtest_portfolio(self, weights)
        period_total = perf['total_return'] if perf else np.nan
        period_annual = perf['annual_return'] if perf else np.nan

        return {
            'type': '사용자지정',
            'label': label,
            'weights': weights,
            'return': stats['return'],   # (모형기대) 연간 기대수익률
            'std': stats['std'],
            'sharpe': stats['sharpe'],
            'period_total_return': period_total,     # 해당 기간 실제 총수익률
            'period_annual_return': period_annual,   # 해당 기간 연환산
            'period_start': self.stock_data.index[0].date(),
            'period_end': self.stock_data.index[-1].date(),
        }

# =========================
# 보조 인터랙션/메뉴 함수들
# =========================
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

    # 효율적 경계선 계산 및 시각화
    print("\n효율적 경계선 계산 및 시각화...")
    efficient_portfolios = optimizer.efficient_frontier(50)
    optimizer.plot_efficient_frontier(efficient_portfolios, optimal_portfolios)

    # 결과 저장
    optimizer.save_results(optimal_portfolios)

    return optimal_portfolios, efficient_portfolios

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
    """사용자 포트폴리오 분석 (자산 전부에 대해 %)"""
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

    max_weight = input("개별 종목 최대 비중 (%, 기본값: 40): ").strip()
    try:
        constraints['max_weight'] = float(max_weight) / 100 if max_weight else 0.4
    except ValueError:
        constraints['max_weight'] = 0.4

    min_weight = input("개별 종목 최소 비중 (%, 기본값: 5): ").strip()
    try:
        constraints['min_weight'] = float(min_weight) / 100 if min_weight else 0.05
    except ValueError:
        constraints['min_weight'] = 0.05

    print(f"\n설정된 제약조건:")
    print(f"  개별 종목 최대 비중: {constraints['max_weight']:.1%}")
    print(f"  개별 종목 최소 비중: {constraints['min_weight']:.1%}")

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
        portfolios, _ = main_with_optimizer(optimizer)
        optimizer.save_results(portfolios)
    elif choice == '2':
        equal_weights = np.array([1/optimizer.n_assets] * optimizer.n_assets)
        equal_portfolio = {'weights': equal_weights, 'type': '균등가중'}
        equal_portfolio.update(optimizer.portfolio_stats(equal_weights))
        optimizer.save_results([equal_portfolio])
    elif choice == '3':
        print("효율적 경계선 계산 중...")
        efficient_portfolios = optimizer.efficient_frontier(100)

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

def parse_selection_dict_line(line, asset_names):
    """
    '삼성전자=50,KB금융=30,SK텔레콤=20' 형태를 dict로 파싱
    자산명이 데이터에 없으면 무시/경고
    """
    result = {}
    if not line:
        return result
    parts = [p.strip() for p in line.split(',') if p.strip()]
    for p in parts:
        if '=' not in p:
            print(f"[무시] 잘못된 항목: {p}")
            continue
        a, w = p.split('=', 1)
        a = a.strip()
        try:
            w = float(w.strip())
        except ValueError:
            print(f"[무시] 비중 숫자 아님: {p}")
            continue
        if a not in asset_names:
            print(f"[무시] 데이터에 없는 종목: {a}")
            continue
        result[a] = w
    return result

def analyze_my_portfolio(optimizer, efficient_portfolios, optimal_portfolios, mc_results):
    """
    사용자 임의 조합을 받아 그래프에 빨간 X 표시 + 기간 수익률 출력
    """
    if efficient_portfolios is None or len(efficient_portfolios) == 0:
        print("효율적 경계선이 없어 새로 계산합니다...")
        efficient_portfolios = optimizer.efficient_frontier(50)

    if mc_results is None or len(mc_results) == 0:
        print("몬테카를로가 없어 새로 계산합니다...")
        mc_results = optimizer.monte_carlo_simulation(5000)

    print("\n나만의 포트폴리오 입력")
    print("예) 삼성전자=50,KB금융=30,SK텔레콤=20  (비중은 % 또는 0~1 모두 허용, 합계는 자동 정규화)")
    print(f"사용 가능 종목: {optimizer.asset_names}")
    line = input("한 줄로 입력: ").strip()
    selection = parse_selection_dict_line(line, optimizer.asset_names)

    if not selection:
        print("입력이 비어 있거나 유효하지 않습니다. 취소합니다.")
        return efficient_portfolios, optimal_portfolios, mc_results

    label = input("라벨(기본: 내 조합): ").strip() or "내 조합"

    try:
        user_point = optimizer.user_portfolio_point(selection, label=label)
    except Exception as e:
        print(f"계산 실패: {e}")
        return efficient_portfolios, optimal_portfolios, mc_results

    # 그래프에 빨간 X 표시
    optimizer.plot_efficient_frontier(efficient_portfolios, optimal_portfolios, highlight=user_point)
    optimizer.plot_monte_carlo(mc_results, highlight=user_point)

    # 콘솔에 기간 수익률 수치로 출력
    print("\n=== 사용자 조합 기간 성과 ===")
    print(f"기간: {user_point['period_start']} ~ {user_point['period_end']}")
    if user_point['period_total_return'] == user_point['period_total_return']:
        print(f"기간 총수익률: {user_point['period_total_return']:.2%}")
        print(f"연환산 수익률: {user_point['period_annual_return']:.2%}")
    else:
        print("기간 수익률 계산 불가(데이터 부족).")
    print(f"(모형기대) 연간 기대수익률: {user_point['return']:.2%}, "
          f"변동성: {user_point['std']:.2%}, 샤프: {user_point['sharpe']:.4f}")

    return efficient_portfolios, optimal_portfolios, mc_results

def interactive_mode():
    """대화형 모드"""
    optimizer = PortfolioOptimizer()

    efficient_portfolios = None
    optimal_portfolios = []
    mc_results = None

    while True:
        print("\n" + "="*50)
        print("        포트폴리오 최적화 도구")
        print("="*50)
        print("1. 데이터 로드")
        print("2. 전체 최적화 실행")
        print("3. 개별 최적화 선택")
        print("4. 사용자 포트폴리오 분석(전 종목 입력)")
        print("5. 효율적 경계선 계산")
        print("6. 몬테카를로 시뮬레이션")
        print("7. 제약조건 설정")
        print("8. 결과 내보내기")
        print("9. 나만의 포트폴리오 리뷰(빨간 X 표시)")
        print("10. 종료")
        print("="*50)

        choice = input("선택하세요 (1-10): ").strip()

        if choice == '1':
            optimizer.load_data()
            if optimizer.stock_data is not None:
                optimizer.calculate_returns_stats()

        elif choice == '2':
            if optimizer.stock_data is None:
                print("먼저 데이터를 로드하세요.")
                continue
            optimal_portfolios, efficient_portfolios = main_with_optimizer(optimizer)

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
            if optimizer.stock_data is None:
                print("먼저 데이터를 로드하세요.")
                continue
            # 필요 시 자동 보충 계산
            if efficient_portfolios is None:
                efficient_portfolios = optimizer.efficient_frontier(50)
            if mc_results is None:
                mc_results = optimizer.monte_carlo_simulation(5000)
            efficient_portfolios, optimal_portfolios, mc_results = analyze_my_portfolio(
                optimizer, efficient_portfolios, optimal_portfolios, mc_results
            )

        elif choice == '10':
            print("프로그램을 종료합니다.")
            break

        else:
            print("잘못된 선택입니다. 1-10 중에서 선택해주세요.")

def main():
    """메인 실행 함수 (자동 실행)"""
    print("="*60)
    print("         포트폴리오 최적화 엔진")
    print("="*60)

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

    # 1. 최소분산
    print("\n1. 최소분산 포트폴리오 계산...")
    min_var = optimizer.min_variance_portfolio()
    if min_var:
        optimal_portfolios.append(min_var)
        optimizer.analyze_portfolio(min_var['weights'], min_var['type'])

    # 2. 최대 샤프
    print("\n2. 최대 샤프비율 포트폴리오 계산...")
    max_sharpe = optimizer.max_sharpe_portfolio()
    if max_sharpe:
        optimal_portfolios.append(max_sharpe)
        optimizer.analyze_portfolio(max_sharpe['weights'], max_sharpe['type'])

    # 3. 제약조건부
    print("\n3. 제약조건부 최적화 계산...")
    constrained = optimizer.optimize_with_constraints({'max_weight': 0.4, 'min_weight': 0.05})
    if constrained:
        optimal_portfolios.append(constrained)
        optimizer.analyze_portfolio(constrained['weights'], constrained['type'])

    # 4. 위험균등
    print("\n4. 위험균등 포트폴리오 계산...")
    risk_parity = optimizer.risk_parity_portfolio()
    if risk_parity:
        optimal_portfolios.append(risk_parity)
        optimizer.analyze_portfolio(risk_parity['weights'], risk_parity['type'])

    # 5. 균등가중(비교)
    equal_weights = np.array([1/optimizer.n_assets] * optimizer.n_assets)
    equal_portfolio = {'weights': equal_weights, 'type': '균등가중'}
    equal_portfolio.update(optimizer.portfolio_stats(equal_weights))
    optimal_portfolios.append(equal_portfolio)
    optimizer.analyze_portfolio(equal_weights, "균등가중")

    # 효율적 경계선
    print("\n효율적 경계선 계산 중...")
    efficient_portfolios = optimizer.efficient_frontier(50)

    # 몬테카를로
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

    valid_portfolios = [p for p in optimal_portfolios if p and p['sharpe'] > 0]
    sorted_portfolios = sorted(valid_portfolios, key=lambda x: x['sharpe'], reverse=True)

    print("샤프비율 기준 포트폴리오 순위:")
    for i, portfolio in enumerate(sorted_portfolios, 1):
        print(f"{i}. {portfolio['type']}: 샤프비율 {portfolio['sharpe']:.4f}, "
              f"수익률 {portfolio['return']:.2%}, 위험 {portfolio['std']:.2%}")

    print(f"\n💰 1000만원 투자 시뮬레이션 (3년 후 예상 금액):")
    initial_investment = 10_000_000
    years = 3
    for portfolio in sorted_portfolios[:5]:
        annual_return = portfolio['return']
        final_amount = initial_investment * (1 + annual_return) ** years
        profit = final_amount - initial_investment
        print(f"  {portfolio['type']:12s}: {final_amount:>12,.0f}원 "
              f"(+{profit:>8,.0f}원, {annual_return:.1%}/년)")

    if sorted_portfolios:
        print(f"\n🎯 추천 포트폴리오: {sorted_portfolios[0]['type']}")
    print("="*60)

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

"""
=== 포트폴리오 최적화 엔진 사용 가이드 ===

📦 필요한 라이브러리:
pip install pandas numpy matplotlib scipy openpyxl

🚀 실행 방법:
1) 자동 실행: python portfolio_optimizer.py
2) 대화형: 실행 후 2번 선택

🧭 '나만의 포트폴리오 리뷰' 사용법(대화형 9번):
- 예: 삼성전자=50,KB금융=30,SK텔레콤=20
- 비중은 % 또는 0~1 모두 허용 (합계는 자동 정규화)
- 그래프에 빨간 X로 표시되고, 콘솔에 기간 수익률이 출력됩니다.
"""
