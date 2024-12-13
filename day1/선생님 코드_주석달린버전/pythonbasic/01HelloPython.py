# 라인단위 주석은 #(샵)을 이용한다. 
'''
파일명 : 01HelloPython.py
블럭단위 주석은 싱글쿼테이션 3개를 연결해서 사용한다. 
'''

'''문장을 먼저 작성합니다. 
그리고 이 부분을 블럭으로 감싼후 싱글을 3번 입력해도 됩니다.'''

# 파이썬은 문자의 끝에 ;(세미콜론)을 사용하지 않는다.
print("Hello python!")

# 한줄에 여러 문장을 기술하고 싶다면 구분을 위해 사용한다. 
print("한줄에  ");print("여러줄 쓰려면 ");print("세미콜런이 필요함")

# 여러 변수 선언
'''좌측항은 변수, 우측항은 할당할 값으로 구분하여 선언 및 초기화된다.
또한 자료형을 기술하지 않아도된다. 값이 초기화될때 자료형은 자동으로
결정된다.'''
r, g, b ="Red", "Green", "Blue"
#여러개의 변수를 선언 및 출력시에는 콤마를 사용한다. 
print(r, g, b)
print(id(r)) # 객체의 주소가 변수에 저장된다.즉 실제값이 저장되지 않고 실제값이 저장된 주소를 기억하는 참조형 변수가 된다.
# 파이썬 변수는 방대한 자료, 함수 그리고 차트 등 다양한 유형의 객체를 담을 수 있는 막강한 역할을 한다.

## 정수형
#파이썬은 변수 선언시 별도의 데이터형을 기술하지 않는다.
x = 2
y = 4
z = 8

# 나누기. 항상 float형 결과를 반환한다. 
print("x / y = ", x/y)   #결과 : 0.5
print("z / y = ", z/y)  #결과 : 2.0
# 나누기. 소수부분을 제거하므로 항상 int형의 결과를 반환한다. (몫)
print("x // y = ", x//y) 
# 곱셈
print("x * y = ", x*y) 
# 거듭제곱. 2의 4승을 계산해서 출력한다. 
print("x ** y = ", x**y) 
# 거듭제곱을 함수를 통해 실행한다. 2의 4승을 계산한다.
print("pow(x, y) = ", pow(x,y))
# 2의 4승을 8로 나눈 나머지를 반환한다. 
print("pow(x, y, z) = ", pow(x,y,z))
# x를 y로 나눈 몫과 나머지를 튜플(배열)로 반환한다.
print("divmod(x, y) = ", divmod(x,y))

# import는 모듈을 불러올때 사용하는 명령으로 math모듈을 사용한다는
# 의미이다. Java에서는 class라고 표현하지만 Python에서는 모듈이라
# 표현한다. 
import math
# 팩토리얼(순차곱셈) 함수이므로 5*4*3*2*1(5!)의 결과를 반환한다.
print("math.factorial(5) =", math.factorial(5)) 
# factorial(5) = 5*factorial(4) 와 같다.

## String 형
str="""아래와 같이
여러줄에 걸쳐 문자열을 작성하고 싶으면
이와 같이 더블쿼테이션 3개 작성한다.
"""
print(str)

head="나는 헤더 "
bottom = " 나는 보텀"
#문자열 합치기
print(head+bottom)
print(head * 3)

#문자열 슬라이싱 : 인덱스와 범위를 통해 문자열을 잘라낼수 있다.
engStr = "Hello Python Good"
#0번 인덱스 : H
print(engStr[0])
#0~3까지의 범위에서 3 앞까지만 가져온다. 즉 0~2까지이므로 : Hel
print(engStr[:3])
#1~2까지만 슬라이싱 한다. : el
print(engStr[1:3])
#1~끝까지 슬라이싱 한다. 
print(engStr[1:])

#Python에서는 한글도 인덱스를 통해 정확히 슬라이싱 할수있다. 
korstr = "안녕하세요? 파이썬입니다."
print(korstr[0])
print(korstr[:2]) #안녕
print(korstr[:6]) #안녕하세요?

'''
format()
    : 문자열을 format()함수를 사용하면 좀 더 발전된 스타일로 문자열
    포맷을 지정할 수 있다. 
    형식] format(0번인덱스, 1번, 2번,....)
        사용시에는 {인덱스}와 같이 사용한다. 
'''
print("{0}는 중복되지 않는 숫자 {1}개로 구성된다.".format("Lotto", 6))
print("{1}는 중복되지 않는 숫자 {0}개로 구성된다.".format("Lotto", 6))

# 인덱스 대신 변수를 사용하는 방법으로 default값을 지정하는 경우
# "변수명=값"으로 사용한다. 
menu1 = "치킨"
menu2 = "맥주"
print("오늘 {str}은 {0}과 {1}로 정했다.".format(menu1, menu2, str='저녁'))