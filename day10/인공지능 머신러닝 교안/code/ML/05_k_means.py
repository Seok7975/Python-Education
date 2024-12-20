# -*- coding: utf-8 -*-
""" 5. K-Means"""

import os # 경고 대응
os.environ['OMP_NUM_THREADS'] = '1' # 스레드 갯수를 한개로 정해준다는 것 넘파이 보다 먼저 실해해야 한다. 오류가 났을 때 실행. 커널에서 리스다트로 해 줄것.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('./data/KMeansData.csv')
dataset[:5]

X = dataset.iloc[:, :].values
# X = dataset.values # 이렇게 해도 됨.
# X = dataset.to_numpy() # 공식 홈페이지 권장
X[:5]

"""### 데이터 시각화 (전체 데이터 분포 확인)"""

plt.scatter(X[:, 0], X[:, 1]) # x축 : hour, y축 : score
plt.title('Score by hours')
plt.xlabel('hours') # 공부시간
plt.ylabel('score') # 점수
plt.show()

"""### 데이터 시각화 (축 범위 통일)

"""

plt.scatter(X[:, 0], X[:, 1]) # x축 : hour, y축 : score
plt.title('Score by hours')
plt.xlabel('hours')
plt.xlim(0, 100) # X 축
plt.ylabel('score')
plt.ylim(0, 100) # Y 축. 두 축이 다르면 실제 구하는 거리와 보이는 거리가 다르기 때문에 두축을 같은 크기로 하는 것이 좋다.
plt.show()

"""### 피처 스케일링 (Feature Scaling)"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X[:5] # 초기값보다 작아지는 것을 볼수 있다.

"""### 데이터 시각화 (스케일링된 데이터)"""

plt.figure(figsize=(5, 5)) # 정확한 사이즈를 위해서 가로 세로 크기를 5로 지정한다. 그럼 정사각형 그래프가 나온다.
plt.scatter(X[:, 0], X[:, 1])
plt.title('Score by hours')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

"""### 최적의 K 값 찾기 (엘보우 방식 Elbow Method)"""

from sklearn.cluster import KMeans # KMeans 클래스 가져오기
inertia_list = [] # 이너시아 fk는게 클러스터에 속한 데이터가 얼마나 가깝게 모이는지를 나타내 주는 것.
for i in range(1, 11): # 클러스터가 1부터 10가지 변하는
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0, n_init='auto')
    # k-means++ 개선된 방법으로 디센트로이드 점을  잡게 된다.
    # n_init='auto'는 랜덤 초기화를 수행하는 횟수를 데이터의 크기와 클러스터 개수에따라 자동으로 결정하는 방식
    kmeans.fit(X) # 학습
    inertia_list.append(kmeans.inertia_) # 각 지점으로부터 클러스터의 중심(centroid) 까지의 거리의 제곱의 합

plt.plot(range(1, 11), inertia_list)
plt.title('Elbow Method')
plt.xlabel('n_clusters') # 클러스터 개수
plt.ylabel('inertia') # 거리의 제곱합이 되는 인너시아
plt.show()

"""### 최적의 K (4) 값으로 KMeans 학습"""

K = 4 # 최적의 n_clusters=K 값

kmeans = KMeans(n_clusters=K, random_state=0, n_init='auto')
# kmeans.fit(X)
y_kmeans = kmeans.fit_predict(X)

y_kmeans

"""### 데이터 시각화 (최적의 K)"""

centers = kmeans.cluster_centers_ # 클러스터의 중심점 (centroid) 좌표
centers

for cluster in range(K): # 0부터 K 까지 반복
    plt.scatter(X[y_kmeans == cluster, 0], X[y_kmeans == cluster, 1], s=100, edgecolor='black') # 각 데이터 형식 => (X축, Y축, 사이즈, 테두리색)
    plt.scatter(centers[cluster, 0], centers[cluster, 1], s=300, edgecolor='black', color='yellow', marker='s') # 클러스터 중심점 내용 형식 => (X좌표, Y좌표, 배경색, 색깔, 마커는 네모속성)
    plt.text(centers[cluster, 0], centers[cluster, 1], cluster, va='center', ha='center') # 클러스터 텍스트 출력 형식 => (X좌표, Y좌표, 보여줄 텍스트(cluster), 세로 위치, 가로위치)

plt.title('Score by hours')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

"""### 데이터 시각화 (스케일링 원복)"""

X_org = sc.inverse_transform(X) # Feature Scaling 된 데이터를 다시 원복
X_org[:5]

centers_org = sc.inverse_transform(centers) # 중심점 좌표 원복
centers_org

# 원복된 데이터의 시각화
for cluster in range(K):
    plt.scatter(X_org[y_kmeans == cluster, 0], X_org[y_kmeans == cluster, 1], s=100, edgecolor='black') # 각 데이터
    plt.scatter(centers_org[cluster, 0], centers_org[cluster, 1], s=300, edgecolor='black', color='yellow', marker='s') # 중심점 네모
    plt.text(centers_org[cluster, 0], centers_org[cluster, 1], cluster, va='center', ha='center') # 클러스터 텍스트 출력

plt.title('Score by hours')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

"""0번째 그룹 같은 경우에는 공부시간은 적은데 상대적으로 점수가 높은 친구들이죠, 즉 공부를 잘하는 친구들이므로 이친구들에게는 다른 과목을 좀더 공부하라고 가이드 할수 있으며, 2번째 클러스터는 공부시간은 굉장히 많은데 점수가 낮습니다. 그럼 이친구들에게 더 효율적인 공부를 할수 있게 지도해야 하고, 1번째 크럴스터는 공부시간도 많고 점수도 좋으므로 일반적인 학새ㅑㅇ이므로 이친구들은 공부시간을 잘ㅃ게 할수 있는 방법을 연구할수 있게하는등의 지도를 할수 있다."""