'''
if 문의 형식
: Java와 동일하지만 여러개의 조건이 필요한 경우 else if가 아닌
elif 를 사용한다. 
형식]
    if 조건1:
        실행문1
    elif 조건2:
        실행문2
    else:
        실행문3
'''
age = 35
'''
print()문은 기본적으로 줄바꿈 기능이 포함되어 있으므로 만약 
줄바꿈 없이 출력하고 싶다면 end=''를 사용하면 된다.
'''
print(age, "살은 ", end="")
if age >= 35:
    print("중년입니다.")
elif age <= 18:
    print("아직 청년이 아닙니다.")
else:
    print("청년 입니다.")
print("="*30)

'''
input() : Java의 Scanner 클래스와 마찬가지로 사용자로부터 입력을
    받을때 사용한다. 
'''
num1 = int(input("숫자 하나를 입력하세요 : "))
if num1 % 2 ==0:
    if num1%3==0:
        print("2와 3으로 나누어짐")
    else:
        print("2는 가능, 3은 불가")
else:
    if num1%3==0:
        print("2는 불가, 3은 가능")
    else:
        print("2와 3 모두 불가")
print("="*30)