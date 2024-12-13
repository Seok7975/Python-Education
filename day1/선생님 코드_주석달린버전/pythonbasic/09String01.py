'''
f-String
: 출력할 문자열 앞에 접두어 f를 붙이고 변수를 {}로 표현한다. 
형식]
    '{문자열변수:<길이}'
    < : 좌측정렬
    > : 우측정렬
    ^ : 센터정렬
    남은 부분은 공백으로 채운다.
'''
str = 'coffee'
num = 1
result = f'{str}는 하루에 {num}잔만 마시는게 좋아요'
print(result)

# 정해진 공간에서의 정렬. (아래에서 사용한 |는 서식이 아니라 단순히
# 공간을 확인하기 위한 용도로 사용되었다.)
s1 = '좌측정렬'
# 10칸의 공간을 확보한 후 문자열을 좌측정렬로 출력한다. 
result1 = f'|{s1:<10}|'
print(result1)

s2 = '센터정렬'
result2 = f'|{s2:^10}|'
print(result2)

s3 = '우측정렬'
result3 = f'|{s3:>10}|'
print(result3)

#f-string에 딕셔너리를 사용 : key를 통해 value를 출력한다. 
dics = {'name':'홍길동', 'gender':'남자', 'age':10}
result = f'{dics["name"]}은 {dics["gender"]}이고 나이는 {dics["age"]}살이다.'
print(result)

#리스트를 사용 : index를 통해 접근한다.
lists = [10, 20, 30]
print(f'리스트 : {lists[0]}, {lists[1]}, {lists[2]}')
for v in lists:
    print(f'반복 출력 : {v}')