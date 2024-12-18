'''
서식문자로 문자열, 정수, 실수를 표현한다. 
    형식] '문자열 서식문자' % (변수 혹은 문자열)
    서식문자는 Java와 동일하고, 자리수를 지정할 경우 
    %10d, %5.3f와 같이 사용할 수 있다. 
'''
# 문자열을 서식문자로 대체해서 사용
str = '내 이름은 %s 입니다.' % '칸'
print("str1=", str)

# 리스트의 크기만큼 반복해서 출력한다.
names = ['유미', '간우', '쟝비']
for n in names:
    print('이름 : %s' % n)

# 정수 : %d
money = 10000
str = '마우스 가격은 %d원 입니다.' % money
print(str)

# 실수 : %f
pi = 3.141592
print('원주율은 %f' % pi)

# 문자열 : %s . 2개 이상의 변수는 콤마로 구분한다.
str = '이름 : %s. 나이 : %d' % ('홍길동', 99)
print(str)

# 여러개의 변수를 초기화시 좌측항과 우측항으로 나눠서 기술한다.
phone, age, height = "010-1234-5678", 28, 181.5
str2 = '전환번호:%s, 나이:%d, 키:%5.2f' % (phone, age, height)
print("str2=", str2)