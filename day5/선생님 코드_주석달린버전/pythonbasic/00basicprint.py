# print : 한개 혹은 여러개 변수를 ,(콤마)로 구분, 
#         출력시 값 사이에 공백이 생김. 터미널에서 해볼 것
print(1, 2, 3)
print('Hello', 'Pyton')

# sep 로 값 사이에 문자 넣기
print(1,2,3, sep=', ')   # sep에 콤마와 공백을 지정 
print(1,2,3, sep=',')   # sep에 콤마만 지정
print(1920, 1080, sep='x')    # sep에 x를 지정

# 줄바꿈 활용
print(1, 2, 3, sep='\n')     # 여러줄에 값을 출력하고 싶다면 sep와 \n을 사용한다.
print('1\n2\n3')    # 문자열 안에 \n을 사용하여 줄바꿈
print("="*60)

# end 로 한줄에 출력하기
'''
 print()를 사용하면 기본적으로 줄바꿈이 된다. 반복문 안에서 print()를 사용할 때
 줄바꿈하지 않고 한줄로 표현하고 싶다면 end 를 사용하면 된다. 
'''
print(1)
print(2)
print(3)

print(1, end='')
print(2, end='')
print(3)
