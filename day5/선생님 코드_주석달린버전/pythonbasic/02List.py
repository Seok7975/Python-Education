'''
파일명 : 02List.py

파이썬에서는 연속된 Collection데이터 구조에 
list, tuple, dictionary, set이 있다. 

리스트(list)
    : 대괄호[] 안에 콤마로 항목을 구분한다. 
    대입연산자로 항목을 변경할 수 있는 시퀀스 자료형이다.
    서로 다른 자료형의 항목으로도 구성할 수 있다. 
    인덱싱, 슬라이싱, 연결, 반복 등이 가능하다. 
    Mutable 데이터 타입이라고 한다. 즉 변경가능한 자료라는 뜻이다.
'''
#기본적인 선언과 사용법은 배열과 동일하다. 
#정수형 리스트와 스트링형 리스트를 선언한다. 
List1 = [1,2,3,4,5]
List2 = ['Java','HTML','Python']
#중첩된 리스트로 선언한다.
List3 = [10,20,['Java','HTML','Python']]

#출력
#리스트 전체가 출력된다. 
print('List1 : ', List1)
#Python이 출력된다.
print('List2[2] : ', List2[2])
#리스트내의 리스트가 출력된다. 
print('List3[2] : ', List3[2])

# 리스트 슬라이싱
print("===슬라이싱", "="*30)
# 1~3까지 슬라이싱된다. 
print('List1[1:4] :', List1[1:4])
# 0~2까지 슬라이싱된다 
print('List1[:3] :', List1[:3])
# 1부터 끝까지 슬라이싱된다. 
print('List1[1:] :', List1[1:])
# List2 0~2까지 이므로 정상적으로 출력된다. 
print('List2[:3] :', List2[:3])
#슬라이싱의 경우 인덱스를 벗어나더라도 에러가 발생하지 않는다.
print('List2[:4] :', List2[:4]) #정상출력
print('List2[:5] :', List2[:5]) #정상출력

#인덱싱의 경우에는 인덱스를 벗어나면 에러가 발생한다. 
# print('List2[5] :', List2[5])  #=> index out of range : 오류발생

print("===리스트 연결", "="*30)
#리스트 내에 또다른 리스트를 삽입하여 연결하는 형태로 사용할수있다.
all_List = [List1, List2]
#2개의 리스트가 1개의 중첩된 리스트로 합쳐진다. 
print("all_List : ", all_List)  # 2차원 배열
print("all_List[1][0] : ", all_List[1][0])  #Java가 출력된다.

print("===List 관련 메소드", "="*30)
# 추가 : 리스트의 마지막 부분에 원소를 추가한다.
List1.append(6)
print('append(6) =>',List1)

# 항목전체를 삭제한다.
List1.clear()
print('clear=>', List1)

List1.extend([10,20,30,40,50])
print('extend=>',List1)

# 삽입 : 1번 인덱스에 99를 삽입한다. 
List1.insert(1,99)
print('insert=>',List1)

# 마지막 원소 반환 및 삭제
print(List1.pop()) # 50이 삭제됨과 동시에 출력된다. 
print('pop()=>',List1)  # 50이 삭제된 리스트를 출력한다.

# 원소삭제 : 처음 나오는 원소 99를 삭제한다.
List1.remove(99)
print("remove=>", List1)

# 리스트 뒤집기
List1.reverse()
print('reverse=>',List1)

'''
List Comprehension(리스트 내포)
    : 대괄호 안에 for루프로 반복적인 표현식을 실행해서 리스트
    원소들을 초기화 하는 방법이다. 
    형식]
        [표현식 for 원소 in 컬렉션 [if조건식]]
'''
print("===List Comprehension", "="*30)
'''
   표현식(n의 제곱) 반복문(0~9까지반복) 조건식(3의배수)
   => 0~9까지의 정수 중 3의 배수의 제곱을 리스트에 초기화한다.
'''
List = [n**2 for n in range(10) if n%3==0]
print(List)