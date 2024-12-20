# -*- coding: utf-8 -*-
# Section 7. 실습 예제
###**7-1. 실습 예제 1~3**


import numpy as np

"""**<예제 1>** 원소가 모두 3인 (3, 4, 5) 형태의 numpy.array를 출력하시오."""

# 답안 작성 (제한 시간 : 20초)
arr = np.full((3, 4, 5), 3)
print(arr)

"""**<예제 2>** 정수 -50 ~ 50의 범위 안의 난수로 이루어진 (4, 5) 형태의 numpy.array를 출력하고 행을 기준으로 오름차순 정렬한 결과와 전체 배열을 1차원 배열로 변경하여 오름차순 정렬한 결과를 출력하시오."""

# 답안 작성 (제한 시간 : 60초)
# array 작성
arr = np.random.randint(-50, 50, (4, 5))
print(arr)
# 행을 기준으로 정렬한 결과 작성
print(np.sort(arr, axis=0))
# 1차원 배열로 변경하여 정렬한 결과 작성
print(np.sort(arr, axis=None))

"""**<예제 3>** 다음과 같은 파이썬 list가 존재한다. list안에 있는 각 numpy.array의 원소들의 평균값과 표준편차, 중앙값을 순서대로 구하여 구한 순서대로 원소가 이루어진 새로운 list를 구성하고 출력하시오.

- 예시 : 각 배열의 평균값이 3.0, 4.0, 5.0이고 표준편차가 1.5, 1.7, 1.9이며 중앙값이 1.0, 2.0, 3.0 이라면 배열로 구성하여 [3.0, 1.5, 1.0, 4.0, 1.7, 2.0, 5.0, 1.9, 3.0] 으로 출력한다.
"""

py_list = [
    np.full(3, 8),
    np.array([33, -15, 26]),
    np.linspace(17, 26, 3)
]

# 답안 작성 (제한 시간 : 50초)
result_arr = []
for i in py_list:
  result_arr.append(np.mean(i))
  result_arr.append(np.std(i))
  result_arr.append(np.median(i))
print(result_arr)

"""###**7-2. 실습 예제 4~6**

**<예제 4>** 다음과 같은 numpy.array가 존재한다. 이 배열을 행을 기준으로 3개의 배열로 분할하여 분할된 각 배열의 원소들을 제곱한 결과를 다시 원본 배열에 행을 기준으로 병합하시오. (단, 마지막 출력 결과는 원본 배열과 차원이 같아야 한다)
"""

arr = np.arange(2, 20, 2).reshape((3, 3))
print(arr)

# 답안 작성 (제한 시간 : 90초)
s1 = np.vsplit(arr, 3)
print(s1)
s2 = np.square(s1)
print(s2)
s3 = np.squeeze(s2, axis=1)
print(s3)
result_arr = np.vstack((arr, s3))
print(result_arr)

"""**<예제 5>** 삼각함수의 특수각(0deg, 30deg, 60deg, 90deg)을 numpy.array로 생성한 후 특수각에 해당하는 sin, cos, tan 값을 각각 구하여 파이썬 list에 담은 다음 해당 list에 들어있는 값들을 출력하시오. (단, 값이 무한대라면 "INF" 문자열을 출력할 것)"""

# 답안 작성 (제한 시간 : 120초)
# numpy의 삼각함수는 radian 값을 사용하기 때문에 degree를 radian으로 변경해야합니다.<degree * PI / 180 >
arr = np.arange(0, 91, 30)
print(arr)

lst = []
lst.append(np.sin(arr * np.pi / 180))
lst.append(np.cos(arr * np.pi / 180))
lst.append(np.tan(arr * np.pi / 180))

for value_lst in lst:
  for value in value_lst:
    if value > 999999999:
      print("INF")
      continue;
    print(value)
  print()

"""**<예제 6>** numpy.array를 이용하여 다음과 같은 패턴을 출력하시오. (단, 출력 시 반복문을 사용하여 출력한다)


---


0 1 0 1 0 1 0   
1 0 1 0 1 0 1   
0 1 0 1 0 1 0   
1 0 1 0 1 0 1   
0 1 0 1 0 1 0   
1 0 1 0 1 0 1   
0 1 0 1 0 1 0
"""

# 답안 작성 (제한 시간 : 120초)
arr = np.zeros((7, 7), dtype=int)
#arr = np.full((7, 7), 0) # 동일한 배열이 출력됨. 둘중에 아무거나 사용
print(arr)

arr[::2, 1::2] = 1 # 홀수 행
arr[1::2, ::2] = 1 # 짝수 행

for row in range(7):
  for col in range(7):
    print(arr[row, col], end=" ")
  print()

"""**<예제 7>** 다음 두 행렬에 대하여 내적 연산을 수행한 결괏값을 출력하시오. (단, 결과의 소수점 아래는 제거한다)"""

arr1 = np.array([[2.1, 3.5],
                 [4.2, 2.7],
                 [2.3, 1.9]])
arr2 = np.array([[5, 2, 3],
                 [1, 3, 5]])
# 답안 작성 (제한 시간 : 60초)
print(np.trunc(np.dot(arr1,arr2)))

"""**<예제 8>** 조건 연산자를 활용한 Boolean 인덱싱을 이용하여 다음 배열의 원소들 중 2와 5의 배수만 추출한 결과를 오름차순 정렬하여 (2, 4) 행렬로 출력하시오.<br/>
![화면 캡처 2024-08-21 165712.png](data:image/png;base64,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)
"""

arr = np.array([[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12]])
print(arr)
# 답안 작성 (제한 시간 : 120초)
print(np.sort(np.append(arr[arr % 2 == 0], arr[arr % 5 == 0])).reshape(2, 4))

"""**<예제 9>** 10진수 100이상 150미만 사이ㅏ에 존재하는 정수를 무작위로 추출하여 (3, 10)형태의 행렬로 만들고 행과 열을 전치한 결과를 출력하시오."""

# 답안 작성 (제한 시간 : 60초)
arr = np.random.randint(low=100, high=150, size=(3, 10))
print(np.transpose(arr))
print(arr.T)
# arr.T, np.transpose(arr) 둘중에 아무거나 해도 됨.

"""**<예제 10>** 10진수 10~20 사이에 존재하는 실수형의 수를 무작위로 10000개 추출하여 100개의 구간에 그래프로 시각화하시오.(단, 무작위 실수 값은 균등한 비율로 추출해야 하며 그래프 시각화 도구는 pyplot 모듈을 사용할 것)"""

# 답안 작성 (제한 시간 : 120초)
import matplotlib.pyplot as plt
randoms = 10 + np.random.rand(10000) * 10
plt.hist(randoms, bins=100)
plt.show()