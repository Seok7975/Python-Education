'''
집합(Set)
: 객체를 참조하기 위한 순서가 없는 Collection으로 원소를 쉽게
추가, 삭제 할 수 있다. 
key가 없는 딕셔너리와 같다. 따라서 중괄호로 선언한다.  
'''
#선언1 : set() 함수를 통해 새로운 set을 생성할 수 있다.
empty_set = set()
print(empty_set)

#선언2 : set()의 인수로 리스트를 전달하여 초기화한다. 
even_set = set([0,2,4,6,8])
print(even_set)

#선언3 : 생성과 동시에 초기화한다. 단, 중복값은 제거된다. 
odd_set = {1,3,5,7,9,7,5,3,1}
print(odd_set)

#새로운 set 초기화
myset = {1, 3, 5}
print(myset)
#추가 : 새로운 원소를 추가한다. 
myset.add(7)
print("myset1 = ", myset)
#여러개의 원소를 추가한다.
myset.update({4, 6, 8})
print("myset2 = ", myset)

#삭제. 단, 여러개를 삭제할 수 없다. 
myset.remove(1)
print("myset3 = ", myset)
#모든 원소를 한꺼번에 삭제한다. 
myset.clear()
print("myset4 = ", myset)

# 집합 연산이 가능하다. 
set_a = {1, 3, 5, 7, 9}
set_b = {1, 2, 5}
print("합집합", set_a | set_b) # 1, 2, 3, 5, 7, 9
print("교집합", set_a & set_b) # 1, 5 
print("차집합", set_a - set_b)  # 3, 7, 9(a에서 b를 뺀 결과를 출력한다.)