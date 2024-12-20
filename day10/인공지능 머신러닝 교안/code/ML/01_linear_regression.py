# -*- coding: utf-8 -*-
"""01. Linear Regression.ipynb"""

import sklearn
sklearn.__version__
# 최신버전으로 업그레이드 후 할 것.

"""### 공부 시간에 따른 시험 점수"""

import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('./data/LinearRegressionData.csv')

dataset.head() # head() - 데이터 상위 5개까지만 보여주는 함수

# 독립변수와 종속 변수의 관계를 인과관계라고 한다. 인과관계는 상관관계에 포함된다.  예) 온도(독립변수)에 의해 수확량(종속변수)이 달라진다.
X = dataset.iloc[:, :-1].values # 처음부터 마지막 컬럼 직전까지의 데이터 (독립 변수 - 원인)
y = dataset.iloc[:, -1].values # 마지막 컬럼 데이터 (종속 변수 - 결과)

X, y

"""# 선형회귀 모델 생성"""

from sklearn.linear_model import LinearRegression
reg = LinearRegression() # 객체 생성
reg.fit(X, y) # 학습 (모델 생성)

y_pred = reg.predict(X) # X 에 대한 예측 값
y_pred

plt.scatter(X, y, color='blue') # 산점도
plt.plot(X, y_pred, color='green') # 예측 선 그래프
plt.title('Score by hours') # 제목
plt.xlabel('hours') # X 축 이름
plt.ylabel('score') # Y 축 이름
plt.show()

print('9시간 공부했을 때 예상 점수 : ', reg.predict([[9]])) # 2차원 배열 형태로 -> [[9], [8], [7]] 9시간 8시간 7시간을 조사할 때

reg.coef_ # 기울기 (m)

reg.intercept_ # y 절편 (b)


import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('./data/LinearRegressionData.csv')
dataset

X = dataset.iloc[:, :-1].values  # 독립변수
y = dataset.iloc[:, -1].values # 종속 변수

from sklearn.model_selection import train_test_split # 테스트 세트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # 훈련 80 : 테스트 20 으로 분리 - test_size=0.2 가 20%를 말함

X, len(X) # 전체 데이터 X, 개수

X_train, len(X_train) # 훈련 세트 X, 개수 -> 80% 이므로 16개

X_test, len(X_test) # 테스트 세트 X, 개수 -> 20% 이므로 4개

y, len(y) # 전체 데이터 y

y_train, len(y_train) # 훈련 세트 y -> 80% 이므로

y_test, len(y_test) # 테스트 세트 y -> 20%

"""### 분리된 데이터를 통한 모델링"""

# 선형회귀 모델
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

reg.fit(X_train, y_train) # 훈련 세트로 학습

"""### 데이터 시각화 (훈련 세트)"""

plt.scatter(X_train, y_train, color='blue') # 산점도
plt.plot(X_train, reg.predict(X_train), color='green') # 선 그래프
plt.title('Score by hours (train data)') # 제목
plt.xlabel('hours') # X 축 이름
plt.ylabel('score') # Y 축 이름
plt.show()

"""### 데이터 시각화 (테스트 세트)"""

plt.scatter(X_test, y_test, color='blue') # 산점도
plt.plot(X_train, reg.predict(X_train), color='green') # 선 그래프 -> 모델을 가지고 만들었기 때문에 그대로 사용
plt.title('Score by hours (test data)') # 제목
plt.xlabel('hours') # X 축 이름
plt.ylabel('score') # Y 축 이름
plt.show()

reg.coef_ # 기울기 -> 위에는 array([10.44369694])

reg.intercept_ # y 절편 -> 위에는 -0.218484702867201

"""### 모델 평가"""

reg.score(X_test, y_test) # 테스트 세트를 통한 모델 평가 -> 97점 정도의 점수

reg.score(X_train, y_train) # 훈련 세트를 통한 모델 평가 -> 93.56점 정도의 점수

"""경사하강법"""

from sklearn.linear_model import SGDRegressor # SGD : Stochastic Gradient Descent 확률적 경사 하강법

sr = SGDRegressor() # 객체 생성
sr.fit(X_train, y_train)

plt.scatter(X_train, y_train, color='blue') # 산점도
plt.plot(X_train, sr.predict(X_train), color='green') # 선 그래프
plt.title('Score by hours (train data, SGD)') # 제목
plt.xlabel('hours') # X 축 이름
plt.ylabel('score') # Y 축 이름
plt.show()

sr.coef_, sr.intercept_
# 주의 : SGDRegressor() 객체를 생성할 때 random_state 값을 지정하지 않았으므로 결과가 다르게 나타날 수 있습니다
# 위에서의 값은 array([10.44369694]) -0.218484702867201

sr.score(X_test, y_test) # 테스트 세트를 통한 모델 평가

sr.score(X_train, y_train) # 훈련 세트를 통한 모델 평가

"""###훈련 회수를 정한 경사 하강법"""

# 지수표기법
# 1e-3 : 0.001 (10^-3)
# 1e-4 : 0.0001 (10^-4)
# 1e+3 : 1000 (10^3)
# 1e+4 : 10000 (10^4)

# 훈련세트를 반복하면서 손실이 어떻게 줄어드는지 보여준다. max_iter 수치를 변경해서 그래프로 확인 할 것.
sr = SGDRegressor(max_iter=100, eta0=1e-4, random_state=0)  # verbose=1 수치를 나타냄.
sr.fit(X_train, y_train)

plt.scatter(X_train, y_train, color='blue') # 산점도
plt.plot(X_train, sr.predict(X_train), color='green') # 선 그래프
plt.title('Score by hours (train data, SGD)') # 제목
plt.xlabel('hours') # X 축 이름
plt.ylabel('score') # Y 축 이름
plt.show()