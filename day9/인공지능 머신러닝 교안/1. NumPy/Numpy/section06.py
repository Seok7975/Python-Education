# -*- coding: utf-8 -*-
import numpy as np

"""# Section 6. N차원 배열의 병합

---

###**6-1. 배열에 원소 추가 및 삭제**
"""

# python list
arr = [1, 2, 3, 4, 5, 6, 7, 8]
arr.insert(2, 50) # 2번째 인덱스에 50 추가
print(arr)

# 1차원 배열에 원소 추가하기
arr = np.arange(1, 9)
arr = np.insert(arr, 2, 50)
print(arr)

# 2차원 배열에 row축을 기준으로 원소 추가하기
arr = np.arange(1, 13).reshape(3, 4)
arr = np.insert(arr, 2, 50, axis=0) # 2번째 행에 50을 추가
print(arr)

# 2차원 배열에 column축을 기준으로 원소 추가하기
arr = np.arange(1, 13).reshape(3, 4)
arr = np.insert(arr, 2, 50, axis=1) # 2번째 열에 50을 추가
print(arr)

# 2차원 배열을 1차원으로 변형하여 원소 추가하기
arr = np.arange(1, 13).reshape(3, 4)
arr = np.insert(arr, 2, 50) # 2번째 인덱스에 50을 추가
print(arr)

# 배열의 특정 원소를 row축을 기준으로 삭제하기
arr = np.arange(1, 13).reshape(3, 4)
print(arr)
arr = np.delete(arr, 2, axis=0) # 2번째 행이 삭제
print(arr)

# 배열의 특정 원소를 column축을 기준으로 삭제하기
arr = np.arange(1, 13).reshape(3, 4)
print(arr)
arr = np.delete(arr, 2, axis=1) # 2번째 열이 삭제
print(arr)

# 2차원 배열을 1차원으로 변형하여 원소 삭제하기
arr = np.arange(1, 13).reshape(3, 4)
print(arr)
arr = np.delete(arr, 2) # 2번째 인덱스 삭제
print(arr)

"""###**6-2. 배열 간의 병합 : append()**"""

# 두 배열을 row축을 기준으로 병합하기
arr1 = np.arange(1, 13).reshape(3, 4)
arr2 = np.arange(13, 25).reshape(3, 4)

print(arr1)
print(arr2, end="\n\n")

arr3 = np.append(arr1, arr2, axis=0) # x 축을 기준으로 병합
print(arr3, end="\n\n")

# 두 배열을 column축을 기준으로 병합하기
arr1 = np.arange(1, 13).reshape(3, 4)
arr2 = np.arange(13, 25).reshape(3, 4)

print(arr1)
print(arr2, end="\n\n")

arr3 = np.append(arr1, arr2, axis=1) # y 축을 기준으로 병합
print(arr3, end="\n\n")

# 두 배열을 1차원 배열로 병합하기
arr1 = np.arange(1, 13).reshape(3, 4)
arr2 = np.arange(13, 25).reshape(3, 4)

print(arr1)
print(arr2, end="\n\n")

arr3 = np.append(arr1, arr2)
print(arr3, end="\n\n")

# vstack(), hstack()
arr1 = np.arange(1, 7).reshape(2, 3)
arr2 = np.arange(7, 13).reshape(2, 3)

arr3 = np.vstack((arr1, arr2)) # x 축을 기준으로 병합
print(arr3, end="\n\n")

arr3 = np.hstack((arr1, arr2)) # y 축을 기준으로 병합
print(arr3, end="\n\n")

# concatenate()
arr1 = np.arange(1, 7).reshape(2, 3)
arr2 = np.arange(7, 13).reshape(2, 3)

arr3 = np.concatenate([arr1, arr2], axis=0) # x 축을 기준으로 병합
print(arr3, end="\n\n")

arr3 = np.concatenate([arr1, arr2], axis=1) # y 축을 기준으로 병합
print(arr3, end="\n\n")

"""###**6-3. 배열 분할**"""

# 두 배열을 row축을 기준으로 분할하기
arr = np.arange(1, 13).reshape(3, 4)
print(arr)

# vsplit() => axis=0
arr_vsplit = np.vsplit(arr, 3) # x 축을 기준때문에 홀수개로 분할 됨.
print(arr_vsplit)

# 두 배열을 column축을 기준으로 분할하기
# hsplit() => axis=1
arr_hsplit = np.hsplit(arr, 2) # y축을 기준때문에 짝수개로 분할 됨.
print(arr_hsplit)

# 3차원 배열 - 텐서
arr = np.random.randint(0, 10, (4, 6, 8))
print(arr)

arr_vsplit = np.vsplit(arr, 2) # x축 기준. 위에 두 부분과 아래 두둡으로 나누어짐
print(arr_vsplit)

arr_hsplit = np.hsplit(arr, 2) # y 축 기준
print(arr_hsplit)