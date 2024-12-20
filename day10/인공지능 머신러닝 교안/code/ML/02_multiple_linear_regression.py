# -*- coding: utf-8 -*-
"""# 2. Multiple Linear Regression (다중 선형 회귀)"""

import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

dataset = pd.read_csv('./data/MultipleLinearRegressionData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X

from sklearn.compose import ColumnTransformer # Column Transformer은 여러 transformer 을 column에 좀 더 쉽게 적용하도록 한 클래스 - 열 변환기
from sklearn.preprocessing import OneHotEncoder # 원핫 인코딩을 위한 클래스
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [2])], remainder='passthrough')
'''
 첫번째 - 'encoder' : 원핫 인코딩을 적용하기 위해 사용
 두번째 - OneHotEncoder : 첫번째 인코딩을 수행할 클래스(원핫인코더) 객체를 넣어준다,
 (drop='first'), [2] : 다중 공정성 문제를 해결하기 위해 첫번째 컬럼을 삭제하고 2개만 사용하겠다.  2 인덱스 컬럼에 적용하겠다.
 remainder='passthrough' : 원하는 코딩을 적용하지 않는 데이터들은 그냥 그대로 둔다.
'''
# 원하는 값을 인코딩 후에 다시 X 에 입력한다.
X = ct.fit_transform(X)
X

# 1 0 : Home
# 0 1 : Library
# 0 0 : Cafe

"""### 데이터 세트 분리"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # 20 %

"""### 학습 (다중 선형 회귀)"""

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

"""### 예측 값과 실제 값 비교 (테스트 세트)"""

y_pred = reg.predict(X_test)
y_pred # 예측 값

y_test # 실제 값

reg.coef_ # 독립변수 (공부장소(집) , 공부장소(도서관), 공부시간, 결석)
# 마이너스면 나쁜 결과 즉 집에서 공부하는게 가장 나쁜 결과 카페는 0이므로
# 공부시간은 1시간씩 증가할때마다 10.4점씩 올라가고, 결석은 1번 할때마다
# 1.64점씩 나쁜 영향을 준다.

reg.intercept_ # y 절편

"""### 모델 평가"""

reg.score(X_train, y_train) # 훈련 세트 - 96.2점

reg.score(X_test, y_test) # 테스트 세트 - 98.59점

"""### 다양한 평가 지표 (회귀 모델)"""

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred) # 실제 값, 예측 값 # MAE

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred) # MSE

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred, squared=False) # RMSE. squared=False -> 제곱하지 말라는 뜻

from sklearn.metrics import r2_score
r2_score(y_test, y_pred) # R2 . 선형회귀 모델평가가 이 R2 스코어 로 계산 된다.