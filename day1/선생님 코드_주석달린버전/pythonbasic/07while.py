'''
반복문
: 파이썬에서의 반복문은 while, for문만 있다. 
do~while문은 없다. 또한 반복문에서 else 구문을 사용할 수 있다.
'''
str = 'python'
'''str이 공백이 될때까지 반복한다. 파이썬에서는 문자열을 배열과
같이 인덱스로 접근할 수 있으므로 아래의 조건은 True가 된다.'''
while str:
    #출력문 끝에 스페이스를 추가하여 줄바꿈없이 출력한다.
    print(str, end=' ')
    #슬라이싱을 통해 제일 앞의 문자 하나를 제거한후 대입한다.
    str= str[1:]
    #python부터 시작하여 마지막 n까지 출력한후 while문을 탈출한다. 
print()
print("="*30)

# 시나리오] 1~10까지의 합을 구하시오. 
sum = 0
'''파이썬에서는 while문에 else를 추가할 수 있다. while문의 
조건이 False일때 탈출하여 else구문을 실행한다. '''
i = 1
while i <= 10:
    sum += i    #누적해서 더하는 부분
    if i<10:
        print(i, end=" + ")
    else:
        print(i, end=" = ")
    i += 1  #i값 증가
else:
    print(sum)
print("="*30)

# 시나리오] 구구단을 출력하되 짝수단만 출력하시오
dan=2
while dan <= 9:
    if dan % 2 == 1:    #홀수단일때..
        dan += 1
        #반복문에서 continue는 처음으로 돌아간다.
        continue
    else:   #짝수단일때 구구단을 출력한다. 
        su=1
        while su <= 9:
            #printf()와 같이 서식문자을 통해 값을 출력한다. 
            #2칸을 확보한 후 변수값을 출력한다.
            print("%2d * %2d = %2d" % (dan, su, su* dan), end=' ')
            su += 1
    dan += 1
    print()
print()

'''
연습문제] while 문으로 피라미드를 출력하시오. 
    *
    **
    ***
    ****
    *****
'''
print("피라미드 출력하기")
x, y = 1, 1
while x <= 5:
    while y <= 5:
        if x>=y:
            print('*', end='')
        y += 1
    print('')
    x += 1
    y=1