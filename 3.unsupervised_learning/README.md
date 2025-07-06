순서	주제	모델	난이도	설명
1단계	군집화	KMeans	⭐	가장 기초 군집화
2단계	군집화	DBSCAN	⭐⭐	밀도 기반 군집화, 이상값 탐지
3단계	군집화	계층적 군집화 (Hierarchical Clustering)	⭐⭐	트리 구조 군집화
4단계	차원 축소	PCA	⭐⭐	고차원 → 저차원 축소, 시각화
5단계	차원 축소	t-SNE	⭐⭐⭐	비선형 데이터 시각화
6단계	이상 탐지	Isolation Forest, One-Class SVM	⭐⭐⭐	이상치 탐지 알고리즘

✅ 추천 학습 흐름
KMeans 군집화
👉 비지도 학습 입문, 핵심 원리 이해

DBSCAN
👉 밀도 기반, 이상치 처리 이해

계층적 군집화
👉 덴드로그램 시각화, 데이터 계층 구조 이해

PCA (주성분 분석)
👉 차원 축소, 데이터 압축, 데이터 시각화

t-SNE
👉 비선형 차원 축소, 고급 시각화

이상 탐지 (Anomaly Detection)
👉 비지도 학습의 응용: 이상값 탐지, 보안 활용

✅ 각 주제별 대표 예제
주제	데이터셋 추천	예제
KMeans	iris, digits	군집 개수에 따른 시각화
DBSCAN	iris, noisy data	이상치 탐지 및 군집 시각화
계층적 군집화	iris	덴드로그램 시각화
PCA	digits	2차원 차원 축소 및 군집 시각화
t-SNE	digits	비선형 차원 축소 시각화
이상 탐지	생성 데이터, 신용카드 이상 거래	Isolation Forest 시각화

✅ 요약
순서	모델	데이터셋	목적
1	KMeans	iris	군집화 기초
2	DBSCAN	iris	밀도 기반 군집화
3	계층적 군집화	iris	계층적 군집화 시각화
4	PCA	digits	차원 축소 및 시각화
5	t-SNE	digits	고급 비선형 시각화
6	Isolation Forest	생성 데이터	이상치 탐지
