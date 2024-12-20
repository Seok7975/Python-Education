# -*- coding: utf-8 -*-
import numpy as np

"""# Section 4. N차원 배열 정렬

---

###**4-1. 1차원 배열의 정렬**
"""

arr = np.random.randint(10, size=10)
print(arr)

# 1차원 배열 오름차순 정렬하기
print(np.sort(arr))

# 1차원 배열 내림차순 정렬하기
print(np.sort(arr)[::-1]) # 처음부터 끝까지 마이너스 한칸 간격으로 인덱싱 해주는 것

# 값을 유지 하려면
arr1 = np.random.randint(10, size=10)
print(arr1)
arr1 = np.sort(arr1) # 변수에 새로 할당 해 주거나
print(arr1)

arr2 = np.random.randint(10, size=10)
print(arr2)
arr2.sort() # 재정의 된 sort()함수를 이용해서 출력
print(arr2)

"""###**4-2. 2차원 배열의 정렬**"""

arr = np.random.randint(15, size=(3, 4))
print(arr)

# 2차원 배열 row 기준 정렬하기 : 기본 정렬임.
print(np.sort(arr))

# 2차원 배열 column 기준 정렬하기
print(np.sort(arr, axis=0))

# 1차원 배열로 변경하여 전체 배열 정렬하기
print(np.sort(arr, axis=None))

# 배열을 정렬할 index 출력하기 : argsort()
arr = np.random.randint(15, size=(3, 4))
print(arr)
print(np.sort(arr, axis=1))
print(np.argsort(arr, axis=1))  # 원래 자기 행의 위치의 인덱스를 출력

# 배열을 정렬할 index 출력하기 : argsort()
arr = np.random.randint(15, size=(3, 4))
print(arr)
print(np.sort(arr, axis=0))
print(np.argsort(arr, axis=0))  # 원래 자기 열의 위치의 인덱스를 출력