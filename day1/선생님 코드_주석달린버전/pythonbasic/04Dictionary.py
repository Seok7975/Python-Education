'''
파일명 : 04Dictionary.py
  
딕셔너리(Dictionary)  
: Map컬렉션과 비슷하고, 순서는 중요하지 않으나 :(콜론)으로
구분되는 고유키(Key)와 값(Value)으로 지정된다. 값은 변경할 수 있고
중괄호로 선언한다. 
'''
#생성1 : dict() 함수를 이용해서 딕셔너리를 생성한다.
dic1 = dict(birth=1870, name='홍길도', size='160')
print(dic1)
#생성2 : 중괄호를 이용해서 생성한다. 
dic2 = {'one':1, "two":2, 'three':3}
print(dic2)

#반복문을 이용한 딕셔너리 출력
fruits = {'apple':100, 'grapes':200, 'orange':300, 'peach':400}
print('for 문을 이용한 출력')
#key를 먼저 얻어온 후..
for key in fruits:
    # 얻어온 key를 통해 value를 출력한다. 
    val = fruits[key]
    #Java의 printf와 유사한 형태의 출력방식으로 서식문자를 사용한다.
    print('%s : %d' % (key, val))

#딕셔너리의 크기를 반환한다. 
print("길이", len(fruits))
print("복숭아", fruits['peach'])    #400출력

#key가 동일하면 기존의 값을 변경한다. 
fruits['orange'] = 3500
print("오렌지", fruits['orange'])
print(fruits)

#key에 해당하는 값을 삭제한다.
del fruits['peach']
print("fruits=", fruits)

#keys() : 딕셔너리의 키로 된 dict_keys 객체를 반환한다. 
get_keys = fruits.keys()
for k in get_keys:
    # Key를 출력한다
    print(k)

#values() : 딕셔너리의 값들로 된 dict_values 객체를 반환한다.
get_values = fruits.values()
for v in get_values:
    # Value를 출력한다. 
    print(v)