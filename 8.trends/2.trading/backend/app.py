# app.py - Flask 백엔드
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
import json
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # React와 연동을 위한 CORS 설정

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
        
    def load_data(self, filename):
        """데이터 로드 (pkl, csv, xlsx 지원)"""
        try:
            file_path = os.path.join(self.data_dir, filename)
            
            if filename.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    self.stock_data = pickle.load(f)
            elif filename.endswith('.csv'):
                self.stock_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            elif filename.endswith('.xlsx'):
                self.stock_data = pd.read_excel(file_path, index_col=0, parse_dates=True)
            else:
                raise ValueError("지원하지 않는 파일 형식입니다.")
            
            # 데이터 전처리
            self.stock_data = self.stock_data.fillna(method='ffill').dropna()
            self.n_assets = len(self.stock_data.columns)
            self.asset_names = list(self.stock_data.columns)
            
            return True
            
        except Exception as e:
            print(f"데이터 로드 실패: {e}")
            return False
    
    def calculate_returns_stats(self):
        """수익률 통계 계산"""
        if self.stock_data is None:
            return False
        
        # 일간 수익률 계산
        self.returns = self.stock_data.pct_change().dropna()
        
        # 연간 기대수익률 (252 거래일)
        self.mu = self.returns.mean() * 252
        
        # 연간 공분산 행렬
        self.sigma = self.returns.cov() * 252
        
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
            'return': float(portfolio_return),
            'std': float(portfolio_std),
            'variance': float(portfolio_variance),
            'sharpe': float(sharpe_ratio)
        }
    
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
            return {
                'weights': result.x.tolist(),
                'weights_dict': dict(zip(self.asset_names, result.x.tolist())),
                **stats,
                'type': '최대샤프'
            }
        return None
    
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
            return {
                'weights': result.x.tolist(),
                'weights_dict': dict(zip(self.asset_names, result.x.tolist())),
                **stats,
                'type': '최소분산'
            }
        return None
    
    def efficient_frontier(self, num_portfolios=50):
        """효율적 경계선 계산"""
        min_var = self.min_variance_portfolio()
        if not min_var:
            return []
        
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
                    'weights': result.x.tolist(),
                    'weights_dict': dict(zip(self.asset_names, result.x.tolist())),
                    **stats
                })
        
        return efficient_portfolios
    
    def monte_carlo_simulation(self, num_simulations=1000):
        """몬테카를로 시뮬레이션"""
        np.random.seed(42)
        results = []
        
        for _ in range(num_simulations):
            weights = np.random.random(self.n_assets)
            weights = weights / np.sum(weights)
            
            stats = self.portfolio_stats(weights)
            results.append({
                'weights': weights.tolist(),
                'weights_dict': dict(zip(self.asset_names, weights.tolist())),
                **stats
            })
        
        return results
    
    def analyze_two_stock_combinations(self):
        """2종목 조합 분석"""
        combinations = []
        corr_matrix = self.returns.corr()
        
        for i in range(self.n_assets):
            for j in range(i + 1, self.n_assets):
                stock1 = self.asset_names[i]
                stock2 = self.asset_names[j]
                
                weights = np.zeros(self.n_assets)
                weights[i] = 0.5
                weights[j] = 0.5
                
                stats = self.portfolio_stats(weights)
                correlation = corr_matrix.iloc[i, j]
                
                combinations.append({
                    'stocks': [stock1, stock2],
                    'correlation': float(correlation),
                    'diversification_benefit': float(1 - abs(correlation)),
                    'weights': weights.tolist(),
                    'weights_dict': dict(zip(self.asset_names, weights.tolist())),
                    **stats
                })
        
        return sorted(combinations, key=lambda x: x['diversification_benefit'], reverse=True)
    
    def get_correlation_matrix(self):
        """상관관계 행렬 반환"""
        if self.returns is None:
            return None
        
        corr_matrix = self.returns.corr()
        return {
            'matrix': corr_matrix.values.tolist(),
            'columns': corr_matrix.columns.tolist(),
            'index': corr_matrix.index.tolist()
        }
    
    def get_individual_stats(self):
        """개별 종목 통계"""
        if self.mu is None or self.sigma is None:
            return []
        
        individual_stats = []
        for i, asset in enumerate(self.asset_names):
            individual_stats.append({
                'name': asset,
                'return': float(self.mu.iloc[i]),
                'risk': float(np.sqrt(self.sigma.iloc[i, i])),
                'sharpe': float(self.mu.iloc[i] / np.sqrt(self.sigma.iloc[i, i]))
            })
        
        return individual_stats

# 전역 옵티마이저 인스턴스
optimizer = PortfolioOptimizer()

@app.route('/api/files', methods=['GET'])
def get_available_files():
    """사용 가능한 데이터 파일 목록 반환"""
    try:
        data_dir = 'data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        files = []
        for ext in ['.pkl', '.csv', '.xlsx']:
            files.extend([f for f in os.listdir(data_dir) if f.endswith(ext)])
        
        return jsonify({
            'success': True,
            'files': files
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/load-data', methods=['POST'])
def load_data():
    """데이터 로드"""
    try:
        data = request.json
        filename = data.get('filename')
        
        if not filename:
            return jsonify({
                'success': False,
                'error': '파일명이 필요합니다.'
            }), 400
        
        success = optimizer.load_data(filename)
        if not success:
            return jsonify({
                'success': False,
                'error': '데이터 로드에 실패했습니다.'
            }), 500
        
        # 수익률 통계 계산
        optimizer.calculate_returns_stats()
        
        return jsonify({
            'success': True,
            'message': f'{filename} 로드 완료',
            'data_info': {
                'filename': filename,
                'assets': optimizer.asset_names,
                'n_assets': optimizer.n_assets,
                'start_date': optimizer.stock_data.index[0].strftime('%Y-%m-%d'),
                'end_date': optimizer.stock_data.index[-1].strftime('%Y-%m-%d'),
                'data_shape': optimizer.stock_data.shape
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/optimize', methods=['POST'])
def optimize_portfolio():
    """포트폴리오 최적화 실행"""
    try:
        if optimizer.stock_data is None:
            return jsonify({
                'success': False,
                'error': '먼저 데이터를 로드하세요.'
            }), 400
        
        data = request.json
        options = data.get('options', {})
        
        # 기본 옵션 설정
        num_simulations = options.get('num_simulations', 1000)
        num_frontier_points = options.get('num_frontier_points', 50)
        
        # 최적화 실행
        results = {}
        
        # 1. 최적 포트폴리오들
        results['optimal_portfolios'] = {}
        
        max_sharpe = optimizer.max_sharpe_portfolio()
        if max_sharpe:
            results['optimal_portfolios']['max_sharpe'] = max_sharpe
        
        min_variance = optimizer.min_variance_portfolio()
        if min_variance:
            results['optimal_portfolios']['min_variance'] = min_variance
        
        # 균등가중 포트폴리오
        equal_weights = [1/optimizer.n_assets] * optimizer.n_assets
        equal_stats = optimizer.portfolio_stats(equal_weights)
        results['optimal_portfolios']['equal_weight'] = {
            'weights': equal_weights,
            'weights_dict': dict(zip(optimizer.asset_names, equal_weights)),
            **equal_stats,
            'type': '균등가중'
        }
        
        # 2. 효율적 경계선
        results['efficient_frontier'] = optimizer.efficient_frontier(num_frontier_points)
        
        # 3. 몬테카를로 시뮬레이션
        results['monte_carlo'] = optimizer.monte_carlo_simulation(num_simulations)
        
        # 4. 2종목 조합 분석
        results['two_stock_analysis'] = optimizer.analyze_two_stock_combinations()
        
        # 5. 추가 정보
        results['correlation_matrix'] = optimizer.get_correlation_matrix()
        results['individual_stats'] = optimizer.get_individual_stats()
        
        # 6. 메타데이터
        results['metadata'] = {
            'assets': optimizer.asset_names,
            'optimization_time': datetime.now().isoformat(),
            'num_simulations': num_simulations,
            'num_frontier_points': num_frontier_points
        }
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/analyze-portfolio', methods=['POST'])
def analyze_custom_portfolio():
    """사용자 포트폴리오 분석"""
    try:
        if optimizer.stock_data is None:
            return jsonify({
                'success': False,
                'error': '먼저 데이터를 로드하세요.'
            }), 400
        
        data = request.json
        weights = data.get('weights', [])
        
        if len(weights) != optimizer.n_assets:
            return jsonify({
                'success': False,
                'error': f'가중치 개수가 맞지 않습니다. {optimizer.n_assets}개 필요.'
            }), 400
        
        # 가중치 정규화
        weights = np.array(weights)
        if abs(np.sum(weights) - 1.0) > 1e-6:
            weights = weights / np.sum(weights)
        
        # 포트폴리오 분석
        stats = optimizer.portfolio_stats(weights)
        
        # 위험 기여도 계산
        portfolio_vol = stats['std']
        marginal_contrib = np.dot(optimizer.sigma, weights) / portfolio_vol
        risk_contrib = weights * marginal_contrib
        
        result = {
            'weights': weights.tolist(),
            'weights_dict': dict(zip(optimizer.asset_names, weights.tolist())),
            **stats,
            'risk_contributions': {
                asset: float(contrib) for asset, contrib in 
                zip(optimizer.asset_names, risk_contrib / np.sum(risk_contrib))
            }
        }
        
        return jsonify({
            'success': True,
            'analysis': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/simulate-investment', methods=['POST'])
def simulate_investment():
    """투자 시뮬레이션"""
    try:
        data = request.json
        portfolio_return = data.get('return', 0)
        initial_amount = data.get('initial_amount', 10000000)
        years = data.get('years', 3)
        
        final_amount = initial_amount * (1 + portfolio_return) ** years
        profit = final_amount - initial_amount
        
        return jsonify({
            'success': True,
            'simulation': {
                'initial_amount': initial_amount,
                'final_amount': final_amount,
                'profit': profit,
                'years': years,
                'annual_return': portfolio_return,
                'total_return': (final_amount / initial_amount) - 1
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """헬스 체크"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'data_loaded': optimizer.stock_data is not None,
        'assets': optimizer.asset_names if optimizer.asset_names else []
    })

# 정적 파일 서빙 (React 빌드 파일용)
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react_app(path):
    if path != "" and os.path.exists(os.path.join('build', path)):
        return send_from_directory('build', path)
    else:
        return send_from_directory('build', 'index.html')

if __name__ == '__main__':
    # data 디렉토리 생성
    if not os.path.exists('data'):
        os.makedirs('data')
    
    print("포트폴리오 최적화 Flask 서버 시작")
    print("사용 가능한 엔드포인트:")
    print("- GET  /api/files - 데이터 파일 목록")
    print("- POST /api/load-data - 데이터 로드")
    print("- POST /api/optimize - 포트폴리오 최적화")
    print("- POST /api/analyze-portfolio - 사용자 포트폴리오 분석")
    print("- POST /api/simulate-investment - 투자 시뮬레이션")
    print("- GET  /api/health - 헬스 체크")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
