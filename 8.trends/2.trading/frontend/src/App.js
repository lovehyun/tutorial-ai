import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ScatterChart, Scatter, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';

const API_BASE_URL = 'http://localhost:5000/api';

const PortfolioApp = () => {
  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState('');
  const [dataInfo, setDataInfo] = useState(null);
  const [optimizationResults, setOptimizationResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('upload');

  // API 호출 함수들
  const fetchFiles = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/files`);
      const data = await response.json();
      if (data.success) {
        setFiles(data.files);
      } else {
        setError(data.error);
      }
    } catch (err) {
      setError('파일 목록을 가져오는데 실패했습니다.');
    }
  };

  const loadData = async (filename) => {
    try {
      setLoading(true);
      setError('');
      
      const response = await fetch(`${API_BASE_URL}/load-data`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ filename }),
      });
      
      const data = await response.json();
      if (data.success) {
        setDataInfo(data.data_info);
        setActiveTab('analysis');
      } else {
        setError(data.error);
      }
    } catch (err) {
      setError('데이터 로드에 실패했습니다.');
    } finally {
      setLoading(false);
    }
  };

  const optimizePortfolio = async () => {
    try {
      setLoading(true);
      setError('');
      
      const response = await fetch(`${API_BASE_URL}/optimize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          options: {
            num_simulations: 1000,
            num_frontier_points: 50
          }
        }),
      });
      
      const data = await response.json();
      if (data.success) {
        setOptimizationResults(data.results);
        setActiveTab('results');
      } else {
        setError(data.error);
      }
    } catch (err) {
      setError('최적화에 실패했습니다.');
    } finally {
      setLoading(false);
    }
  };

  const simulateInvestment = (portfolioReturn, initialAmount = 10000000, years = 3) => {
    const finalAmount = initialAmount * Math.pow(1 + portfolioReturn, years);
    const profit = finalAmount - initialAmount;
    return { finalAmount, profit };
  };

  useEffect(() => {
    fetchFiles();
  }, []);

  // 컴포넌트들
  const FileUploadTab = () => (
    <div className="bg-white p-6 rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-6">📁 데이터 파일 선택</h2>
      
      {files.length === 0 ? (
        <div className="text-center py-8">
          <p className="text-gray-500 mb-4">data 폴더에 CSV, Excel, 또는 PKL 파일을 업로드하세요.</p>
          <div className="bg-blue-50 p-4 rounded-lg">
            <p className="text-sm text-blue-700">
              <strong>지원 형식:</strong> .csv, .xlsx, .pkl<br/>
              <strong>데이터 구조:</strong> 날짜(인덱스) × 종목(컬럼) 형태의 주가 데이터
            </p>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              사용할 데이터 파일을 선택하세요:
            </label>
            <select
              value={selectedFile}
              onChange={(e) => setSelectedFile(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">파일을 선택하세요...</option>
              {files.map((file) => (
                <option key={file} value={file}>
                  {file}
                </option>
              ))}
            </select>
          </div>
          
          <button
            onClick={() => selectedFile && loadData(selectedFile)}
            disabled={!selectedFile || loading}
            className="w-full bg-blue-600 text-white py-3 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            {loading ? '로딩 중...' : '데이터 로드'}
          </button>
        </div>
      )}
      
      <button
        onClick={fetchFiles}
        className="mt-4 text-blue-600 hover:text-blue-800 text-sm underline"
      >
        파일 목록 새로고침
      </button>
    </div>
  );

  const DataAnalysisTab = () => (
    <div className="bg-white p-6 rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-6">📊 데이터 분석</h2>
      
      {dataInfo ? (
        <div className="space-y-6">
          <div className="bg-green-50 p-4 rounded-lg">
            <h3 className="font-semibold text-green-800 mb-2">✅ 데이터 로드 완료</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <p><strong>파일명:</strong> {dataInfo.filename}</p>
                <p><strong>종목 수:</strong> {dataInfo.n_assets}개</p>
                <p><strong>데이터 크기:</strong> {dataInfo.data_shape[0]}일 × {dataInfo.data_shape[1]}종목</p>
              </div>
              <div>
                <p><strong>시작일:</strong> {dataInfo.start_date}</p>
                <p><strong>종료일:</strong> {dataInfo.end_date}</p>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold mb-2">포함된 종목:</h4>
            <div className="flex flex-wrap gap-2">
              {dataInfo.assets.map((asset) => (
                <span
                  key={asset}
                  className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm"
                >
                  {asset}
                </span>
              ))}
            </div>
          </div>

          <button
            onClick={optimizePortfolio}
            disabled={loading}
            className="w-full bg-green-600 text-white py-3 px-4 rounded-md hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            {loading ? '최적화 실행 중...' : '🚀 포트폴리오 최적화 실행'}
          </button>
        </div>
      ) : (
        <div className="text-center py-8">
          <p className="text-gray-500">먼저 데이터를 로드해주세요.</p>
        </div>
      )}
    </div>
  );

  const ResultsTab = () => {
    if (!optimizationResults) {
      return (
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl font-bold mb-6">📈 최적화 결과</h2>
          <div className="text-center py-8">
            <p className="text-gray-500">최적화를 실행해주세요.</p>
          </div>
        </div>
      );
    }

    const { optimal_portfolios, efficient_frontier, monte_carlo, two_stock_analysis, individual_stats, correlation_matrix } = optimizationResults;
    const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#8dd1e1'];

    return (
      <div className="space-y-8">
        {/* 헤더 */}
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-6 rounded-lg">
          <h1 className="text-3xl font-bold mb-2">포트폴리오 최적화 결과</h1>
          <p className="text-lg">효율적 경계선 기반 과학적 투자 전략</p>
        </div>

        {/* 통합 차트 */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl font-bold mb-6 text-center">🎯 5종목 최적 배분 분석: 효율적 경계선</h2>
          
          <ResponsiveContainer width="100%" height={600}>
            <ScatterChart margin={{ top: 20, right: 30, bottom: 100, left: 80 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                type="number" 
                dataKey="std" 
                name="위험도"
                domain={['dataMin - 0.01', 'dataMax + 0.01']}
                tickFormatter={(value) => `${(value * 100).toFixed(1)}%`}
                label={{ value: '연간 위험도 (표준편차)', position: 'insideBottom', offset: -20 }}
              />
              <YAxis 
                type="number" 
                dataKey="return" 
                name="수익률"
                domain={['dataMin - 0.01', 'dataMax + 0.01']}
                tickFormatter={(value) => `${(value * 100).toFixed(1)}%`}
                label={{ value: '연간 기대수익률', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip 
                content={({ active, payload, label }) => {
                  if (active && payload && payload.length > 0) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-white p-4 border rounded-lg shadow-lg">
                        <p className="font-semibold mb-2">
                          {data.type || '포트폴리오'}
                        </p>
                        <p>수익률: {(data.return * 100).toFixed(2)}%</p>
                        <p>위험도: {(data.std * 100).toFixed(2)}%</p>
                        <p>샤프비율: {data.sharpe?.toFixed(3) || 'N/A'}</p>
                        {data.weights_dict && (
                          <div className="mt-2 pt-2 border-t">
                            <p className="font-semibold text-sm">종목 배분:</p>
                            {Object.entries(data.weights_dict).map(([stock, weight]) => (
                              <p key={stock} className="text-xs">
                                {stock}: {(weight * 100).toFixed(1)}%
                              </p>
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Legend />
              
              {/* 몬테카를로 시뮬레이션 (배경, 투명하게) */}
              <Scatter 
                name="랜덤 포트폴리오" 
                data={monte_carlo.slice(0, 300)} 
                fill="#e5e7eb" 
                fillOpacity={0.4}
                r={3}
              />
              
              {/* 효율적 경계선 (빨간 점들) */}
              <Scatter 
                name="효율적 경계선" 
                data={efficient_frontier} 
                fill="#dc2626" 
                fillOpacity={0.8}
                r={4}
              />
              
              {/* 개별 종목들 (정사각형) */}
              <Scatter 
                name="개별 종목" 
                data={individual_stats.map(stock => ({
                  ...stock,
                  symbol: "square"
                }))} 
                fill="#7c3aed" 
                shape="square"
                r={6}
              />
              
              {/* 🎯 최적 포트폴리오들 - 크고 명확한 X 표시 */}
              {optimal_portfolios.max_sharpe && (
                <Scatter 
                  name="⭐ 최대 샤프비율 (최적)" 
                  data={[{
                    ...optimal_portfolios.max_sharpe,
                    symbol: "cross",
                    size: 200
                  }]} 
                  fill="#ffd700" 
                  stroke="#b45309"
                  strokeWidth={3}
                  shape="cross"
                  r={12}
                />
              )}
              {optimal_portfolios.min_variance && (
                <Scatter 
                  name="🛡️ 최소 분산" 
                  data={[{
                    ...optimal_portfolios.min_variance,
                    symbol: "triangle"
                  }]} 
                  fill="#10b981" 
                  stroke="#047857"
                  strokeWidth={2}
                  shape="triangle"
                  r={10}
                />
              )}
              {optimal_portfolios.equal_weight && (
                <Scatter 
                  name="⚖️ 균등 가중" 
                  data={[{
                    ...optimal_portfolios.equal_weight,
                    symbol: "diamond"
                  }]} 
                  fill="#f59e0b" 
                  stroke="#d97706"
                  strokeWidth={2}
                  shape="diamond"
                  r={8}
                />
              )}
              
              {/* 최적 2종목 조합 표시 */}
              {two_stock_analysis.length > 0 && (
                <Scatter 
                  name="🎲 최적 2종목 조합" 
                  data={[{
                    ...two_stock_analysis[0],
                    symbol: "star"
                  }]} 
                  fill="#ec4899" 
                  stroke="#be185d"
                  strokeWidth={2}
                  shape="star"
                  r={10}
                />
              )}
            </ScatterChart>
          </ResponsiveContainer>
          
          {/* 차트 하단 설명 */}
          <div className="mt-4 bg-blue-50 p-4 rounded-lg">
            <h4 className="font-semibold text-blue-800 mb-2">📊 차트 해석 가이드:</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-blue-700">
              <div>
                <p><strong>⭐ 금색 X (최대 샤프비율)</strong>: 위험 대비 수익률이 가장 우수한 5종목 최적 배분</p>
                <p><strong>🛡️ 녹색 삼각형</strong>: 위험을 최소화하는 보수적 배분</p>
                <p><strong>⚖️ 주황 다이아몬드</strong>: 각 종목 20%씩 균등 배분</p>
              </div>
              <div>
                <p><strong>🔴 빨간 점들</strong>: 수학적으로 최적인 위험-수익률 조합들</p>
                <p><strong>🟦 보라 사각형</strong>: 개별 종목 투자 시 위험-수익률</p>
                <p><strong>🎲 분홍 별</strong>: 2종목만 투자할 때 최적 조합</p>
              </div>
            </div>
          </div>
        </div>

        {/* 🎯 최적 배분 상세 분석 */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl font-bold mb-6 text-center">🎯 1,000만원 최적 투자 배분 전략</h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* 최적 포트폴리오 배분 (파이차트) */}
            <div className="bg-gradient-to-br from-yellow-50 to-amber-50 p-6 rounded-lg border-2 border-yellow-200">
              <h3 className="text-xl font-bold text-center mb-4 text-yellow-800">
                ⭐ 최대 샤프비율 포트폴리오 (추천)
              </h3>
              {optimal_portfolios.max_sharpe && (
                <>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={Object.entries(optimal_portfolios.max_sharpe.weights_dict || {}).map(([name, value]) => ({
                          name,
                          value,
                          percentage: (value * 100).toFixed(1),
                          amount: (value * 10000000).toLocaleString()
                        }))}
                        cx="50%"
                        cy="50%"
                        outerRadius={100}
                        fill="#8884d8"
                        dataKey="value"
                        label={({name, percentage}) => `${name}: ${percentage}%`}
                        labelLine={false}
                      >
                        {Object.keys(optimal_portfolios.max_sharpe.weights_dict || {}).map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                        ))}
                      </Pie>
                      <Tooltip 
                        formatter={(value, name, props) => [
                          `${props.payload.percentage}% (${props.payload.amount}원)`,
                          '투자 비중'
                        ]}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                  
                  {/* 상세 배분 테이블 */}
                  <div className="mt-4">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="bg-yellow-100">
                          <th className="p-2 text-left">종목</th>
                          <th className="p-2 text-right">비중</th>
                          <th className="p-2 text-right">투자금액</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(optimal_portfolios.max_sharpe.weights_dict || {})
                          .sort(([,a], [,b]) => b - a) // 비중 순으로 정렬
                          .map(([stock, weight]) => (
                          <tr key={stock} className="border-b">
                            <td className="p-2 font-medium">{stock}</td>
                            <td className="p-2 text-right font-semibold text-blue-600">
                              {(weight * 100).toFixed(1)}%
                            </td>
                            <td className="p-2 text-right text-green-600">
                              {(weight * 10000000).toLocaleString()}원
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  
                  {/* 성과 지표 */}
                  <div className="mt-4 grid grid-cols-3 gap-2 text-center text-sm">
                    <div className="bg-white p-2 rounded">
                      <div className="font-bold text-green-600">
                        {(optimal_portfolios.max_sharpe.return * 100).toFixed(2)}%
                      </div>
                      <div className="text-xs text-gray-600">연간 수익률</div>
                    </div>
                    <div className="bg-white p-2 rounded">
                      <div className="font-bold text-blue-600">
                        {(optimal_portfolios.max_sharpe.std * 100).toFixed(2)}%
                      </div>
                      <div className="text-xs text-gray-600">연간 위험도</div>
                    </div>
                    <div className="bg-white p-2 rounded">
                      <div className="font-bold text-purple-600">
                        {optimal_portfolios.max_sharpe.sharpe?.toFixed(3) || 'N/A'}
                      </div>
                      <div className="text-xs text-gray-600">샤프 비율</div>
                    </div>
                  </div>
                </>
              )}
            </div>

            {/* 최적화 이유 및 전략 설명 */}
            <div>
              <h3 className="text-xl font-bold mb-4">💡 최적화 전략 해석</h3>
              
              <div className="space-y-4">
                {/* 최적화 근거 */}
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-blue-800 mb-2">🔬 수학적 최적화 근거</h4>
                  <ul className="text-sm text-blue-700 space-y-1">
                    <li>• <strong>마코위츠 평균-분산 모델</strong> 적용</li>
                    <li>• <strong>샤프 비율 극대화</strong>: 위험 단위당 수익률 최적화</li>
                    <li>• <strong>공분산 행렬</strong> 기반 종목간 상관관계 고려</li>
                    <li>• <strong>제약조건</strong>: 가중치 합 = 100%, 공매도 금지</li>
                  </ul>
                </div>

                {/* 포트폴리오 특성 */}
                <div className="bg-green-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-green-800 mb-2">🎯 포트폴리오 특성</h4>
                  {optimal_portfolios.max_sharpe && (
                    <ul className="text-sm text-green-700 space-y-1">
                      <li>• <strong>섹터 분산</strong>: IT, 통신, 금융 균형 배분</li>
                      <li>• <strong>위험 조정</strong>: 고위험 종목 비중 제한</li>
                      <li>• <strong>상관관계 활용</strong>: 낮은 상관관계 종목 비중 증대</li>
                      <li>• <strong>유동성 고려</strong>: 대형주 중심 안정적 구성</li>
                    </ul>
                  )}
                </div>

                {/* 투자 시나리오 */}
                <div className="bg-yellow-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-yellow-800 mb-2">📈 투자 시나리오</h4>
                  {optimal_portfolios.max_sharpe && (
                    <div className="text-sm text-yellow-700 space-y-2">
                      <div className="grid grid-cols-3 gap-2 text-center">
                        <div>
                          <div className="font-bold">1년 후</div>
                          <div>{(10000000 * (1 + optimal_portfolios.max_sharpe.return)).toLocaleString()}원</div>
                        </div>
                        <div>
                          <div className="font-bold">3년 후</div>
                          <div className="text-green-600">
                            {simulateInvestment(optimal_portfolios.max_sharpe.return, 10000000, 3).finalAmount.toLocaleString()}원
                          </div>
                        </div>
                        <div>
                          <div className="font-bold">5년 후</div>
                          <div>{simulateInvestment(optimal_portfolios.max_sharpe.return, 10000000, 5).finalAmount.toLocaleString()}원</div>
                        </div>
                      </div>
                      <p className="text-center mt-2">
                        <strong>3년 수익:</strong> 
                        <span className="text-green-600 font-bold ml-1">
                          +{simulateInvestment(optimal_portfolios.max_sharpe.return, 10000000, 3).profit.toLocaleString()}원
                        </span>
                      </p>
                    </div>
                  )}
                </div>

                {/* 리밸런싱 가이드 */}
                <div className="bg-purple-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-purple-800 mb-2">🔄 리밸런싱 가이드</h4>
                  <ul className="text-sm text-purple-700 space-y-1">
                    <li>• <strong>주기</strong>: 분기별 (3개월) 리밸런싱 권장</li>
                    <li>• <strong>기준</strong>: 목표 비중 ±5% 이상 차이 시</li>
                    <li>• <strong>방법</strong>: 비중 높은 종목 매도 → 낮은 종목 매수</li>
                    <li>• <strong>비용</strong>: 거래 수수료 고려한 점진적 조정</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 2종목 다각화 분석 */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl font-bold mb-6">🎲 2종목 투자 다각화 분석</h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h4 className="text-lg font-semibold mb-3">📈 최적 2종목 조합 (상위 5개)</h4>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse border border-gray-300 text-sm">
                  <thead>
                    <tr className="bg-gray-100">
                      <th className="border border-gray-300 p-2">종목 조합</th>
                      <th className="border border-gray-300 p-2">상관계수</th>
                      <th className="border border-gray-300 p-2">다각화효과</th>
                      <th className="border border-gray-300 p-2">샤프비율</th>
                      <th className="border border-gray-300 p-2">순위</th>
                    </tr>
                  </thead>
                  <tbody>
                    {two_stock_analysis.slice(0, 5).map((combo, index) => (
                      <tr key={index} className={index === 0 ? "bg-green-50 font-semibold" : ""}>
                        <td className="border border-gray-300 p-2">
                          {combo.stocks.join(' + ')}
                        </td>
                        <td className="border border-gray-300 p-2">
                          {combo.correlation.toFixed(3)}
                        </td>
                        <td className="border border-gray-300 p-2 text-green-600">
                          {(combo.diversification_benefit * 100).toFixed(1)}%
                        </td>
                        <td className="border border-gray-300 p-2">
                          {combo.sharpe?.toFixed(3) || 'N/A'}
                        </td>
                        <td className="border border-gray-300 p-2 text-center">
                          {index === 0 ? "🥇" : index === 1 ? "🥈" : index === 2 ? "🥉" : index + 1}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* 상관관계 히트맵 */}
            <div>
              <h4 className="text-lg font-semibold mb-3">🔗 종목간 상관관계 매트릭스</h4>
              {correlation_matrix && (
                <div className="bg-gray-50 p-4 rounded-lg overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr>
                        <th className="p-2 text-xs font-semibold text-gray-700 border"></th>
                        {correlation_matrix.columns.map(stock => (
                          <th key={stock} className="p-2 text-xs font-semibold text-gray-700 border text-center min-w-[60px]">
                            {stock.length > 6 ? stock.substring(0, 6) + '...' : stock}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {correlation_matrix.matrix.map((row, i) => (
                        <tr key={i}>
                          <td className="p-2 text-xs font-semibold text-gray-700 border bg-gray-100">
                            {correlation_matrix.index[i].length > 6 ? 
                              correlation_matrix.index[i].substring(0, 6) + '...' : 
                              correlation_matrix.index[i]}
                          </td>
                          {row.map((corr, j) => {
                            const intensity = Math.abs(corr);
                            let bgColor, textColor;
                            
                            if (i === j) {
                              // 대각선 (자기 자신과의 상관관계 = 1)
                              bgColor = '#1f2937';
                              textColor = 'white';
                            } else if (corr > 0.7) {
                              // 강한 양의 상관관계
                              bgColor = `rgba(34, 197, 94, ${0.7 + intensity * 0.3})`;
                              textColor = 'white';
                            } else if (corr > 0.3) {
                              // 중간 양의 상관관계
                              bgColor = `rgba(59, 130, 246, ${0.4 + intensity * 0.4})`;
                              textColor = 'white';
                            } else if (corr > 0) {
                              // 약한 양의 상관관계
                              bgColor = `rgba(59, 130, 246, ${0.2 + intensity * 0.3})`;
                              textColor = 'black';
                            } else if (corr > -0.3) {
                              // 약한 음의 상관관계
                              bgColor = `rgba(239, 68, 68, ${0.2 + intensity * 0.3})`;
                              textColor = 'black';
                            } else {
                              // 강한 음의 상관관계
                              bgColor = `rgba(239, 68, 68, ${0.5 + intensity * 0.5})`;
                              textColor = 'white';
                            }
                            
                            return (
                              <td
                                key={j}
                                className="p-2 text-center text-xs border font-medium"
                                style={{ 
                                  backgroundColor: bgColor,
                                  color: textColor,
                                  minWidth: '60px'
                                }}
                                title={`${correlation_matrix.index[i]} vs ${correlation_matrix.columns[j]}: ${corr.toFixed(3)}`}
                              >
                                {corr.toFixed(2)}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  
                  {/* 범례 */}
                  <div className="mt-4 flex flex-wrap gap-4 text-xs">
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 rounded" style={{backgroundColor: 'rgba(34, 197, 94, 0.8)'}}></div>
                      <span>강한 양의 상관관계 (&gt; 0.7)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 rounded" style={{backgroundColor: 'rgba(59, 130, 246, 0.6)'}}></div>
                      <span>중간 양의 상관관계 (0.3-0.7)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 rounded" style={{backgroundColor: 'rgba(59, 130, 246, 0.3)'}}></div>
                      <span>약한 양의 상관관계 (0-0.3)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 rounded" style={{backgroundColor: 'rgba(239, 68, 68, 0.3)'}}></div>
                      <span>약한 음의 상관관계 (0 to -0.3)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 rounded" style={{backgroundColor: 'rgba(239, 68, 68, 0.8)'}}></div>
                      <span>강한 음의 상관관계 (&lt; -0.3)</span>
                    </div>
                  </div>
                  
                  {/* 해석 도움말 */}
                  <div className="mt-3 p-3 bg-blue-50 rounded text-xs">
                    <p className="font-semibold text-blue-800 mb-1">💡 상관관계 해석:</p>
                    <p className="text-blue-700">
                      • <strong>1.0</strong>: 완전히 같은 움직임 | 
                      • <strong>0.7+</strong>: 매우 유사한 움직임 | 
                      • <strong>0.3-0.7</strong>: 어느 정도 유사 | 
                      • <strong>0-0.3</strong>: 약한 관계 | 
                      • <strong>음수</strong>: 반대 방향 움직임
                    </p>
                    <p className="text-blue-600 mt-1">
                      <strong>다각화 관점:</strong> 낮은 상관관계(0.3 이하)일수록 포트폴리오 위험 분산 효과가 큽니다.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* 최적 2종목 조합 상세 */}
          {two_stock_analysis.length > 0 && (
            <div className="mt-6 bg-green-50 p-4 rounded-lg">
              <h4 className="font-semibold text-green-800 mb-2">
                🏆 최적 2종목 조합: {two_stock_analysis[0].stocks.join(' + ')}
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div>
                  <p><strong>상관계수:</strong> {two_stock_analysis[0].correlation.toFixed(3)}</p>
                  <p><strong>다각화 효과:</strong> {(two_stock_analysis[0].diversification_benefit * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <p><strong>기대수익률:</strong> {(two_stock_analysis[0].return * 100).toFixed(2)}%</p>
                  <p><strong>위험도:</strong> {(two_stock_analysis[0].std * 100).toFixed(2)}%</p>
                </div>
                <div>
                  <p><strong>샤프비율:</strong> {two_stock_analysis[0].sharpe?.toFixed(3) || 'N/A'}</p>
                  <p><strong>3년 후 예상:</strong> {simulateInvestment(two_stock_analysis[0].return).finalAmount.toLocaleString()}원</p>
                </div>
              </div>
              <div className="mt-3 text-green-700">
                <p><strong>선택 이유:</strong></p>
                <p>• 가장 낮은 상관관계로 최고의 다각화 효과</p>
                <p>• 서로 다른 특성으로 시장 리스크 분산</p>
                <p>• 안정성과 성장성의 균형있는 조합</p>
              </div>
            </div>
          )}
        </div>

        {/* 상관관계 상세 분석 */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl font-bold mb-6">🔗 상관관계 상세 분석</h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* 상관관계 차트 */}
            <div>
              <h3 className="text-lg font-semibold mb-3">종목별 상관관계 비교</h3>
              {correlation_matrix && (
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={
                    correlation_matrix.columns.flatMap((stock1, i) => 
                      correlation_matrix.columns.slice(i + 1).map((stock2, j) => ({
                        pair: `${stock1.substring(0, 4)}-${stock2.substring(0, 4)}`,
                        correlation: correlation_matrix.matrix[i][i + j + 1],
                        diversification: 1 - Math.abs(correlation_matrix.matrix[i][i + j + 1])
                      }))
                    ).sort((a, b) => a.correlation - b.correlation)
                  }>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="pair" 
                      angle={-45} 
                      textAnchor="end" 
                      height={80}
                      fontSize={10}
                    />
                    <YAxis 
                      domain={[-1, 1]}
                      tickFormatter={(value) => value.toFixed(1)}
                    />
                    <Tooltip 
                      formatter={(value, name) => [
                        value.toFixed(3),
                        name === 'correlation' ? '상관계수' : '다각화 효과'
                      ]}
                    />
                    <Legend />
                    <Bar 
                      dataKey="correlation" 
                      fill="#8884d8" 
                      name="상관계수"
                    />
                    <Bar 
                      dataKey="diversification" 
                      fill="#82ca9d" 
                      name="다각화 효과"
                    />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>

            {/* 다각화 효과 순위 */}
            <div>
              <h3 className="text-lg font-semibold mb-3">🎯 다각화 효과 순위</h3>
              {correlation_matrix && (
                <div className="space-y-2 max-h-80 overflow-y-auto">
                  {correlation_matrix.columns.flatMap((stock1, i) => 
                    correlation_matrix.columns.slice(i + 1).map((stock2, j) => ({
                      stock1,
                      stock2,
                      correlation: correlation_matrix.matrix[i][i + j + 1],
                      diversification: 1 - Math.abs(correlation_matrix.matrix[i][i + j + 1])
                    }))
                  )
                  .sort((a, b) => b.diversification - a.diversification)
                  .slice(0, 10)
                  .map((item, index) => (
                    <div 
                      key={`${item.stock1}-${item.stock2}`}
                      className={`p-3 rounded-lg border ${
                        index === 0 ? 'bg-green-50 border-green-200' :
                        index === 1 ? 'bg-blue-50 border-blue-200' :
                        index === 2 ? 'bg-yellow-50 border-yellow-200' :
                        'bg-gray-50 border-gray-200'
                      }`}
                    >
                      <div className="flex justify-between items-center">
                        <div>
                          <span className="font-semibold">
                            {index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : `${index + 1}.`}
                          </span>
                          <span className="ml-2 font-medium">
                            {item.stock1} + {item.stock2}
                          </span>
                        </div>
                        <div className="text-right text-sm">
                          <div className="font-semibold text-green-600">
                            다각화: {(item.diversification * 100).toFixed(1)}%
                          </div>
                          <div className="text-gray-600">
                            상관계수: {item.correlation.toFixed(3)}
                          </div>
                        </div>
                      </div>
                      
                      {index < 3 && (
                        <div className="mt-2 text-xs text-gray-600">
                          {index === 0 && "✨ 최고의 다각화 효과 - 가장 독립적인 움직임"}
                          {index === 1 && "⭐ 우수한 다각화 효과 - 위험 분산에 효과적"}
                          {index === 2 && "👍 좋은 다각화 효과 - 포트폴리오 안정성 기여"}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* 상관관계 인사이트 */}
          <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
            {correlation_matrix && (() => {
              const correlations = correlation_matrix.columns.flatMap((stock1, i) => 
                correlation_matrix.columns.slice(i + 1).map((stock2, j) => 
                  correlation_matrix.matrix[i][i + j + 1]
                )
              );
              
              const avgCorr = correlations.reduce((sum, corr) => sum + corr, 0) / correlations.length;
              const maxCorr = Math.max(...correlations);
              const minCorr = Math.min(...correlations);
              
              return (
                <>
                  <div className="bg-blue-50 p-4 rounded-lg text-center">
                    <div className="text-2xl font-bold text-blue-600">{avgCorr.toFixed(3)}</div>
                    <div className="text-sm text-blue-700">평균 상관계수</div>
                    <div className="text-xs text-blue-600 mt-1">
                      {avgCorr > 0.6 ? "높은 상관관계" : 
                       avgCorr > 0.3 ? "중간 상관관계" : "낮은 상관관계"}
                    </div>
                  </div>
                  
                  <div className="bg-green-50 p-4 rounded-lg text-center">
                    <div className="text-2xl font-bold text-green-600">{minCorr.toFixed(3)}</div>
                    <div className="text-sm text-green-700">최저 상관계수</div>
                    <div className="text-xs text-green-600 mt-1">최고 다각화 기회</div>
                  </div>
                  
                  <div className="bg-red-50 p-4 rounded-lg text-center">
                    <div className="text-2xl font-bold text-red-600">{maxCorr.toFixed(3)}</div>
                    <div className="text-sm text-red-700">최고 상관계수</div>
                    <div className="text-xs text-red-600 mt-1">유사한 움직임</div>
                  </div>
                </>
              );
            })()}
          </div>
        </div>
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl font-bold mb-6">📊 개별 종목 분석</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={individual_stats}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="name" 
                angle={-45} 
                textAnchor="end" 
                height={80}
                fontSize={12}
              />
              <YAxis tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
              <Tooltip 
                formatter={(value, name) => [
                  `${(value * 100).toFixed(2)}%`,
                  name === 'return' ? '기대수익률' : 
                  name === 'risk' ? '위험도' : '샤프비율'
                ]}
              />
              <Legend />
              <Bar dataKey="return" fill="#8884d8" name="기대수익률" />
              <Bar dataKey="risk" fill="#82ca9d" name="위험도" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* 투자 시나리오 분석 */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl font-bold mb-6">💰 투자 시나리오 분석 (1,000만원 기준)</h2>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse border border-gray-300">
              <thead>
                <tr className="bg-gray-100">
                  <th className="border border-gray-300 p-3">시나리오</th>
                  <th className="border border-gray-300 p-3">1년 후</th>
                  <th className="border border-gray-300 p-3">3년 후</th>
                  <th className="border border-gray-300 p-3">5년 후</th>
                  <th className="border border-gray-300 p-3">연평균 수익률</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(optimal_portfolios)
                  .filter(([key, portfolio]) => portfolio)
                  .sort((a, b) => (b[1].sharpe || 0) - (a[1].sharpe || 0))
                  .slice(0, 3)
                  .map(([key, portfolio], index) => {
                    const names = {
                      max_sharpe: "최적 포트폴리오 ⭐",
                      min_variance: "최소 분산",
                      equal_weight: "균등 가중"
                    };
                    return (
                      <tr key={key} className={index === 0 ? "bg-yellow-50" : ""}>
                        <td className="border border-gray-300 p-3 font-semibold">{names[key] || portfolio.type}</td>
                        <td className="border border-gray-300 p-3">
                          {(10000000 * (1 + portfolio.return)).toLocaleString()}원
                        </td>
                        <td className="border border-gray-300 p-3 text-green-600 font-semibold">
                          {simulateInvestment(portfolio.return, 10000000, 3).finalAmount.toLocaleString()}원
                        </td>
                        <td className="border border-gray-300 p-3">
                          {simulateInvestment(portfolio.return, 10000000, 5).finalAmount.toLocaleString()}원
                        </td>
                        <td className="border border-gray-300 p-3">
                          {(portfolio.return * 100).toFixed(2)}%
                        </td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </div>
        </div>

        {/* 결론 */}
        <div className="bg-gradient-to-r from-green-600 to-teal-600 text-white p-6 rounded-lg">
          <h2 className="text-2xl font-bold mb-4 text-center">🎯 최종 결론 및 투자 권고</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-lg font-semibold mb-2">📈 5종목 최적 배분</h3>
              <div className="bg-white bg-opacity-20 p-3 rounded text-sm space-y-1">
                {optimal_portfolios.max_sharpe && (
                  <>
                    <p><strong>최적 전략:</strong> 최대 샤프비율 포트폴리오</p>
                    <p><strong>예상 연수익률:</strong> {(optimal_portfolios.max_sharpe.return * 100).toFixed(2)}%</p>
                    <p><strong>위험 수준:</strong> {(optimal_portfolios.max_sharpe.std * 100).toFixed(2)}% (적정)</p>
                    <p><strong>3년 목표:</strong> 1,000만원 → {(simulateInvestment(optimal_portfolios.max_sharpe.return).finalAmount/10000).toFixed(0)}만원</p>
                  </>
                )}
                <p><strong>핵심 장점:</strong> 과학적 최적화로 위험 대비 최고 수익률</p>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-2">🎲 2종목 집중 투자</h3>
              <div className="bg-white bg-opacity-20 p-3 rounded text-sm space-y-1">
                {two_stock_analysis.length > 0 && (
                  <>
                    <p><strong>최적 조합:</strong> {two_stock_analysis[0].stocks.join(' + ')}</p>
                    <p><strong>다각화 효과:</strong> {(two_stock_analysis[0].diversification_benefit * 100).toFixed(0)}% (우수)</p>
                    <p><strong>상관계수:</strong> {two_stock_analysis[0].correlation.toFixed(2)} (낮음)</p>
                  </>
                )}
                <p><strong>관리 편의성:</strong> 높음 (2종목만 관리)</p>
                <p><strong>핵심 장점:</strong> 단순하면서도 효과적인 다각화</p>
              </div>
            </div>
          </div>

          <div className="mt-4 bg-white bg-opacity-20 p-3 rounded">
            <h3 className="text-lg font-semibold mb-2">🛠️ 실행 가이드</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div>
                <p><strong>1단계: 준비</strong></p>
                <p>• 증권계좌 개설</p>
                <p>• 투자 자금 확보</p>
                <p>• 리스크 허용도 점검</p>
              </div>
              <div>
                <p><strong>2단계: 실행</strong></p>
                <p>• 최적 비중대로 매수</p>
                <p>• 분할 매수 고려</p>
                <p>• 시장 타이밍 분산</p>
              </div>
              <div>
                <p><strong>3단계: 관리</strong></p>
                <p>• 분기별 리밸런싱</p>
                <p>• 성과 모니터링</p>
                <p>• 전략 재검토</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="container mx-auto px-4 py-8">
        {/* 네비게이션 */}
        <div className="bg-white rounded-lg shadow-lg mb-8">
          <div className="flex border-b">
            <button
              onClick={() => setActiveTab('upload')}
              className={`px-6 py-3 font-medium ${
                activeTab === 'upload'
                  ? 'border-b-2 border-blue-500 text-blue-600'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              📁 데이터 업로드
            </button>
            <button
              onClick={() => setActiveTab('analysis')}
              className={`px-6 py-3 font-medium ${
                activeTab === 'analysis'
                  ? 'border-b-2 border-blue-500 text-blue-600'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
              disabled={!dataInfo}
            >
              📊 데이터 분석
            </button>
            <button
              onClick={() => setActiveTab('results')}
              className={`px-6 py-3 font-medium ${
                activeTab === 'results'
                  ? 'border-b-2 border-blue-500 text-blue-600'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
              disabled={!optimizationResults}
            >
              📈 최적화 결과
            </button>
          </div>
        </div>

        {/* 에러 메시지 */}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-4">
            <strong>오류:</strong> {error}
          </div>
        )}

        {/* 로딩 표시 */}
        {loading && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white p-6 rounded-lg">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
              <p className="text-lg text-center">처리 중...</p>
            </div>
          </div>
        )}

        {/* 탭 콘텐츠 */}
        {activeTab === 'upload' && <FileUploadTab />}
        {activeTab === 'analysis' && <DataAnalysisTab />}
        {activeTab === 'results' && <ResultsTab />}
      </div>
    </div>
  );
};

export default PortfolioApp;
