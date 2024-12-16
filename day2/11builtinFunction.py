'''
Built-in Function(내장함수)
: 내장함수는 외부모듈과 달리 import가 필요하지 않기 때문에
아무런 설정없이 바로 사용할 수 있다. 
int(), min(), print()와 같은 함수가 있다.
'''
'''
enumerate()
: 순서가 있는 자료형(리스트, 튜플, 문자열)을 입력받아 인덱스를
포함한 객체를 반환한다.      
'''
print("="*5, "enumerate()", "="*20)
data = ['Naver', 'Kakao', 'Google']
#반복시 인덱스를 포함한 값을 반환한다.
for i, v in enumerate(data):
    #인덱스 출력을 위한 별도의 증감식을 기술하지 않아도된다. 
    print(i, v)

'''
eval()
: 실행가능한 문자열을 입력받아 실행한 결과를 반환한다. 
즉 문자열 좌우측의 쿼테이션을 잘라낸 후 내부의 코드를 실행한다. 
'''
print("="*5, "eval()", "="*20)
print(eval('1+2'))
print(eval("'hi' + 'a'"))

'''
sorted()
: 입력값을 정렬한 후 리스트로 반환한다. 
'''
print("="*5, "sorted()", "="*20)
numbers = (8,7,6,8,4,9,7,5,3,2,7,4,9,8,2,6,5)
print(sorted(numbers))

'''
이터레이터(iterator)
: 값을 차례대로 꺼낼수 있는 객체이다. 
for i in range(100): 과 같이 작성하면 파이썬은 0부터 99까지의
값을 차례대로 꺼낼 수 있는 이터레이터를 생성하여 숫자를 생성한다. 

iter() : 반복을 끝낼값(Sentinel)을 지정하면 특정값이 나올때
    반복을 종료하는 함수. 
    형식]   
        iter(반복가능한 객체, 반복을 끝낼 값)
''' 
import sys
print("="*5, "이터레이터 함수1", "="*20)
# 리스트 생성
list=[1,2,3,4]
# 이터레이터 함수에 리스트를 인수로 전달하여 객체를 생성한다.
it = iter(list)
# 무한루프 
while True:
    try:
        #반복할 항목이 있는경우 순서대로 반환한다.
        print('it=',next(it))
    #더이상 반복할 항목이 없다면 예외가 발생한다. 
    except:
        print("예외발생")
        #예외가 발생하면 무한루프를 탈출한다.
        break
print()

print("="*5, "이터레이터 함수2", "="*20)
#난수생성을 위한 모듈 임포트
import random
'''
iter함수의 
    첫번째인수 => lambda : random.randint(0,10)
        0~10까지의 난수를 생성하는 람다식으로 정의됨.
    두번째인수 => 2
        이터레이터를 종료할 값 지정.
'''
for i in iter(lambda : random.randint(0,10), 2):
    print(i, end=' ')
print()

'''
제너레이터 함수
: 제너레이터는 이터레이터를 생성해주는 함수로 연속적인 값을
차례대로 생성할 수 있다. 데이터를 미리 만들어 놓지 않아도 되므로
메모리를 절약할 수 있다. 그리고 return문 대신 yield문을 사용하여
반복가능한 객체를 생성한다. 
''' 
print("="*5, "제너레이터 함수1", "="*20)
print('yield의 동작 과정')
#제너레이터 함수 정의
def number_generator1():
    #첫번째 next()를 통해 0을 함수외부로 반환(양보)한다.
    yield 0
    #next()를 지속적으로 호출하면 순차적으로 yield가 실행되어 값이
    #반환된다.
    yield 1
    yield 2
#제너레이터 함수의 참조값을 변수에 저장한다.
g = number_generator1()
#next()함수를 통해 반복적으로 호출한다. 
print('next(g)', next(g))
print('next(g)', next(g))
print('next(g)', next(g))

'''
for문을 통해 반복할때마다 next()를 자동으로 호출하므로 yield에서
발생시킨 값을 순차적으로 가져올 수 있다. 
'''
print("="*5, "제너레이터 함수2", "="*20)
#제너레이터 함수 생성
def number_genertor2(n):
    i = 0
    #yield가 실행될때 현재 i값을 for문으로 전달하여 출력한다.
    while i < n:
        yield i 
        #i값이 증가하여 반복문을 탈출할 조건이 만들어진다. 
        i += 1
#for문에서 호출하면 next()가 자동으로 호출되어 제너레이터가 실행된다.
for i in number_genertor2(7):
    print(i) #실행결과 : 0 1 2 3 4

print("="*5, "제너레이터 함수3", "="*20)
def upper_genertor3(val):
    for i in val:
        #리스트의 원소를 대문자로 변경한 후 함수밖으로 반환한다.
        yield i.upper()
fruits = ['apple', 'pear', 'grape', 'pineapple', 'orange']
#리스트의 갯수만큼 반복하여 제너레이터를 호출한다. 
#next()는 자동으로 호출된다. 
for i in upper_genertor3(fruits):
    print(i, end=' ')