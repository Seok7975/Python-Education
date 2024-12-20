# -*- coding: utf-8 -*-
"""# 4. Logistic Regression - 분류 모델"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('data/LogisticRegressionData.csv') # 데이터 불러오기
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

"""### 데이터 분리"""

from sklearn.model_selection import train_test_split # 4개의 데이터로 분리됨.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # 20% 분리

"""### 학습 (로지스틱 회귀 모델)"""

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression() # 객체 생성
classifier.fit(X_train, y_train) # 학습

"""### 6시간 공부했을 때 예측?"""

classifier.predict([[6]]) # 6시간 공부했을 때의 예측 2차원 배열로
# 결과 1 : 합격할 것으로 예측

classifier.predict_proba([[6]]) # 합격할 확률 출력
# 불합격 확률 14%, 합격 확률 86%

"""### 4시간 공부했을 때 예측?"""

classifier.predict([[4]]) # 4시간 공부했을 때의 예측 2차원 배열로
# 결과 0 : 불합격할 것으로 예측

classifier.predict_proba([[4]]) # 합격할 확률 출력
# 불합격 확률 62%, 합격 확률 38%

"""### 분류 결과 예측 (테스트 세트)"""

y_pred = classifier.predict(X_test)
y_pred # 예측 값

y_test # 실제 값 (테스트 세트)

X_test # 공부 시간 (테스트 세트)

classifier.score(X_test, y_test) # 모델 평가
# 전체 테스트 세트 4개 중에서 분류 예측을 올바로 맞힌 개수 3개 -> 3/4 = 0.75 맞힌 개수로 점수가 나온다는 것이 다름

"""### 데이터 시각화 (훈련 세트)"""

X_range = np.arange(np.min(X), np.max(X), 0.1) # X 의 최소값에서 최대값까지를 0.1 단위로 잘라서 데이터 생성
X_range

p = 1 / (1 + np.exp(-(classifier.coef_ * X_range + classifier.intercept_))) # y = mx + b  를 대입해서 시그모이드 함수로 변환
p

p.shape # 로우 1개와 컬럼 95개인 2차원 배열

X_range.shape # 95개인 1차원 배열

p = p.reshape(-1) # 2차원 배열을 1차원 배열 형태로 변경. -1인 자동으로 대입. 예) 4X4 = 2X(-1) 이면 -1에 자동으 8이 들어감
p.shape

plt.scatter(X_train, y_train, color='blue')
plt.plot(X_range, p, color='green')
plt.plot(X_range, np.full(len(X_range), 0.5), color='red') # X_range 개수만큼 0.5 로 가득찬 배열 만들기. 합격 가능선
plt.title('Probability by hours') # 합격 가능 시간
plt.xlabel('hours')
plt.ylabel('P')
plt.show()

"""### 데이터 시각화 (테스트 세트)"""

plt.scatter(X_test, y_test, color='blue')
plt.plot(X_range, p, color='green')
plt.plot(X_range, np.full(len(X_range), 0.5), color='red') # X_range 개수만큼 0.5 로 가득찬 배열 만들기
plt.title('Probability by hours (test)')
plt.xlabel('hours')
plt.ylabel('P')
plt.show()

classifier.predict_proba([[4.5]]) # 4.5 시간 공부했을 때 확률 (모델에서는 51% 확률로 합격 예측, 실제로는 불합격)

"""### 혼동 행렬 (Confusion Matrix)"""

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

# TRUE NEGATIVE (TN)=1개       FALSE POSITIVE (FP)=1개
# 불합격일거야 (예측)          합격일거야 (예측)
# 불합격 (실제)                불합격 (실제)

# FALSE NEGATIVE (FN)=0개      TRUE POSITIVE (TP)=2개
# 불합격일거야 (예측)          합격일거야 (예측)
# 합격 (실제)                  합격 (실제)

# 왼쪽위 오른쪽 아래는 올바로 예측한 거고 오른쪽위 왼쪽 아래는 잘못 예측을 한 것이다.