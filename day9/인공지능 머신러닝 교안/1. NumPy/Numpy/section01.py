# -*- coding: utf-8 -*-
import numpy as np

"""# Section 1. N차원 배열 생성


---

###**1-1. N차원 배열(ndarray) 생성하기**
"""



# 1차원 배열 생성하기
arr = np.array([1, 2, 3])
print(arr)

# 2차원 배열 생성하기
arr = np.array([[1, 2, 3],
               [4, 5, 6]])
print(arr)

# 리스트
type([1, 2, 3])

# 넘파이 어래이
type(arr)
# 이두가지를 혼동하지 말것

# list나 tuple로 N차원 배열 생성하기
tpl = (4, 5, 6)
arr = np.array(tpl)
print(arr)

lst = [1, 2, 3]
arr = np.array(lst)
print(arr)

lst = [[1, 2, 3], [4, 5, 6]]
arr = np.array(lst)
print(arr)

# shape 확인하기 - 배열의 형태
arr1 = np.array([1, 2, 3])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])

print(arr1.shape, arr2.shape)

# ndim 확인하기 - 차원 표시
print(arr1.ndim, arr2.ndim)

# size 확인하기 - 크기
print(arr1.size, arr2.size)

"""###**1-2. N차원 배열의 데이터 타입**"""

# 배열의 데이터 타입을 int로 지정하기
arr = np.array([1.1, 2.1, 3.1],dtype=np.int32) # 뒤에 실수가 클수록 큰수를 나타낼수 있음.
print(arr, arr.dtype)

# 배열의 데이터 타입을 float으로 지정하기
arr = np.array([1, 2, 3],dtype=np.float32) # 뒤에 실수가 클수록 큰수를 나타낼수 있음.
print(arr, arr.dtype)

# 배열의 데이터 타입을 bool로 지정하기
arr = np.array([0, 1, 1],dtype=np.bool_)
print(arr, arr.dtype)

# 배열의 데이터 타입을 string으로 지정하기
arr = np.array([0, 1, 1],dtype=np.string_)
print(arr, arr.dtype)

# 배열의 데이터 타입 출력하기 - dtype
arr = np.array([0, 1, 2, 3])
print(arr, arr.dtype)

arr = arr.astype(np.float32) # 실수형으로 변경
print(arr, arr.dtype)

# 데이터 타입이 혼재하는 경우
arr = np.array([1, 2, 3.4, "64"], dtype=int) # "64, 문자열" 이렇게 뭊자열이 들어가면 에러남
print(arr, arr.dtype)

"""###**1-3. 정해진 형식의 N차원 배열 생성하기**"""

# 모든 요소들이 0으로 초기화된 N차원 배열 생성하기 - np.zeros()
arr = np.zeros([2, 2])
print(arr)

# 모든 요소들이 1로 초기화된 N차원 배열 생성하기 - np.ones()
arr = np.ones([3, 5])
print(arr)

# 모든 요소들이 지정된 fill_value로 초기화된 N차원 배열 생성하기 - np.full()
arr = np.full((2, 3), 5)  # fuu((함수인자), 필밸류) 5로 채워지는 배열
print(arr)

# NxN 또는 (NXM) shape를 가진 대각 원소가 1인 행렬 생성하기 - np.eye()
arr = np.eye(3, 4, k=1)  # k값의 의해 대각선 위치가 달라짐. 기본값은 0
print(arr)

arr = np.eye(3)  # 하나만 주어질때는 정방향 행렬이 만들어짐
print(arr)

arr = np.array([[1,2,3],
                [4,5,6]])

# 모든 요소들이 0으로 초기화된 N차원 배열로 변형하기 - zeros_like()
arr_z = np.zeros_like(arr)
print(arr_z)

# 모든 요소들이 1로 초기화된 N차원 배열로 변형하기 - ones_like()
arr_o = np.ones_like(arr)
print(arr_o)

# 모든 요소들이 지정된 fill_value로 초기화된 N차원 배열로 변형하기 - full_like()
arr_f = np.full_like(arr, 9)
print(arr_f)

"""###**1-4. 특정 범위의 값을 가지는 N차원 배열 생성하기**"""

lst = list(range(0, 9, 2))
print(lst)

# 특정 범위의 값을 가지는 N차원 배열을 생성하는 함수 3가지 작성하기
arr = np.arange(9)
print(arr)

arr = np.arange(3, 12)
print(arr)

arr = np.arange(3, 13, 3)
print(arr)

arr = np.arange(stop=9, step=2, start=0) # 인자들을 직접 입력
print(arr)

# np.linspace()
arr = np.linspace(0, 100, 250) # stop 값까지 출력, 마지막은 가지수
print(arr)

arr = np.linspace(1, 10, 10)
print(arr, "\n\n")

# np.logspace()
arr = np.logspace(1, 10, 10, base=2) # 2의 제곱승
print(arr)

arr = np.logspace(1, 10, 10) # 10의 제곱승. 사용로그
print(arr)

"""###**1-5. 난수로 이루어진 N차원 배열 생성하기**"""

# 데이터 시각화 라이브러리 matplotlib을 설치하고 import 하기
# 주피터노트북에서 설치하기 !pip install matplotlib
import  matplotlib.pyplot as plt

# 정규 분포로 표본을 추출하는 난수 생성 함수 작성하기 - np.random.normal()
arr = np.random.normal(0, 1, (2,3))
print(arr)

# random 모듈에 존재하는 함수를 활용하여 표본을 추출하고 데이터를 시각화하기

arr = np.random.normal(0, 1, 1000)  # 표본 개수를 늘리면(100000) 확실하게 알 수 있음.
plt.hist(arr, bins =100)  # bins : 난수를 몇개의 구간으로 구분에서 보여줄지 정하는 것
plt.show()
# 1000개의 난수가 100개의 수을 기준으로 구분해서 보여줌

# 균등 분포로 표본을 추출하는 난수 생성 함수 작성하기 - random.rand()
arr = np.random.rand(1000)  # 표본 개수를 인자로
plt.hist(arr, bins =100)  # bins : 난수를 몇개의 구간으로 구분에서 보여줄지 정하는 것
plt.show()
# 난수가 균등하게 분포되는 것을 볼수 있음.

# randn() : -1 부터 1사이의 값을 정규분포 또는 가오시안 분포의 표본을 추출하는 함수
arr = np.random.randn(1000)
plt.hist(arr, bins =100)
plt.show()
# 난수를 정규분포 형태로 보여줌

# 랜덤한 정수를 생성하는  함수 작성하기 - randint()
arr = np.random.randint(low=1, high=5, size=10)  # 로우값부터 하이값 사이의 값을 배열로 출력
print(arr)

arr = np.random.randint(low=1, high=5, size=(3, 4))  # 3X4 행력 출력
print(arr)

arr = np.random.randint(5)  # 0부터 5사이의 하나의 값이 출력
print(arr)

# 히스토그램을 이용한 출력
arr = np.random.randint(100, 200, 1000)
plt.hist(arr, bins =100)
plt.show()

"""###**1-6. 시드(Seed)값을 통한 난수 생성 제어**"""

arr = np.random.rand(10)
print("난수 발생1 \n", arr)

arr = np.random.rand(10)
print("난수 발생2 \n", arr)

# 시드 값을 이용해 난수의 발생 지점을 고정하기
np.random.seed(1) # 씨드값을 1로 지정
arr = np.random.rand(10)
print("난수 발생1 \n", arr)

np.random.seed(1)
arr = np.random.rand(10)
print("난수 발생1 \n", arr)
 # 씨드값을 1로 지정했기때문에 동일한 값이 나오는 것을 알수 있다. 시작점만 같으면 동일한 난수 표본이 나온다.