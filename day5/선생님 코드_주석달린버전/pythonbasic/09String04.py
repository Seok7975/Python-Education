'''
capitalize()
 : 첫 문자가 대문자이고 나머지가 소문자인 문자열의 복사본을 리턴한다.
'''
a = 'PYTHON'
print(a.capitalize())

'''
count(x, start, end)
    :  인자 x의 갯수를 카운트하여 리턴한다.
    x - 개수를 찾을 문자열 또는 문자입니다.
    str (선택 사항) - 검색이 시작되는 문자열 내에서 시작 색인.
    end (선택 사항) - 검색이 끝나는 문자열 내에서 끝 인덱스.
'''
a = 'pythonpp'
print(a.count('p')) 

a = 'pythjonpy'
print(a.count('py',0,5))  # 0 -> str  5 -> end

'''
find(sub, start, end)
    : 문자열 내에서 하위 문자열의 인덱스를 반환한다.
    문자나 문자열이 없을시 -1 을 반환한다.
    sub - 검색 할 하위 문자열 입니다.
    start (선택 사항) - 검색이 시작되는 문자열 내에서 시작 색인.
    end(선택 사항) - 검색이 끝나는 문자열 내에서 끝 인덱스.
'''
a = 'pythonpy'
print(a.find('y'))
print(a.find('py',0,5))
print(a.find('tk'))

'''
index(sub, start, end)
    : 문자열 내에서 하위 문자열의 인덱스를 반환한다. 문자나 문자열이 
    없을시 오류가 발생한다.
    sub - 검색 할 하위 문자열 입니다.
    start (선택 사항) - 검색이 시작되는 문자열 내에서 시작 색인.
    end(선택 사항) - 검색이 끝나는 문자열 내에서 끝 인덱스.
'''
a = 'pythonpy'
print(a.index('y')) # 결과 : 1 인덱스
print(a.index('py',0,5)) # 결과 : 0
print(a.index('tk'))

'''
isalnum()
     : 문자열의 모든 문자가 숫자 또는 알파벳 또는 한글 이면 True 를 반환,
     아니면 False 를 반환
'''
a = 'python'
print(a.isalnum())  # >>> True

a = 'python홍길동'
print(a.isalnum()) # >>> True

a = 'python%%'
print(a.isalnum())  # >>> False

'''
isalpha()
    : 문자열의 모든 문자가 알파벳이면 True 반환, 아니면 False 반환
    그러나 테스트시 한글이 들어가도 True 반환, 알파벳과 한글이 나뉘어 
    있을시에는 False 반환
'''
a = 'python'
print(a.isalpha())  # 결과 : True

a = 'python홍길동'
print(a.isalpha())  # 결과 : True

a = 'python 홍길동'
print(a.isalpha())  # 결과 : False

a = 'python%%' 
print(a.isalpha())  # 결과 : False

'''
isdecimal()
 : 문자열의 모든 문자가 10진수이면 True 반환, 아니면 False 반환
'''
a = '123'
print(a.isdecimal())    # >>> True 

a = '123 ##'
print(a.isdecimal())    # >>> False

a = '123 홍길동'
print(a.isdecimal())    # >>> False

'''
isdigit()
    : 문자열의 모든 문자가 숫자인 경우 True 반환, 아니면 False 반환
'''
a = '123'
print(a.isdigit())

a = '123##'
print(a.isdigit())

a = '123홍길동'
print(a.isdigit()) 

a = 'python123'
print(a.isdigit())

'''
islower()
    : 문자열의 모든 알파벳이 소문자 이면 True 반환, 문자열에 대문자 하나 
    이상 있으면 False 반환
'''
a = 'python'
print(a.islower()) # 결과 : True 

a = 'Python'
print(a.islower()) # 결과 : False 

'''
isspace()
    : 문자열에 공백 문자열만 있을경우 True 를 반환, 아니면 False 반환
    간격에 사용되는 문자를 공백 문자라고한다. [예 : 탭, 공백, 줄 바꿈 등]
'''
a = '    \t'
print(a.isspace())  # 결과 : True
a = ' a '
print(a.isspace()) # 결과 : False

'''
isupper()
    : 문자열의  모든 알파벳이 대문자 이면 True 반환, 문자열에 소문자 하나 
    이상 있으면 False 반환
'''
a = 'PYTHON'
print(a.isupper()) # 결과 : True 
a = 'PYTHON홍길동'
print(a.isupper()) # 결과 : True 
a = 'PYTHOn%%'
print(a.isupper()) # 결과 : False

'''
lower()
문자열의 모든 대문자를 소문자로 변환합니다.
upper()
문자열의 모든 소문자를 대문자로 변환합니다.
'''
a = 'PYTHON'
print(a.lower())

'''
swapcase()
    : 문자열의 모든 대문자는 소문자로 변환, 문자열의 모든 소문자는 
    대문자로 변환
'''
string = 'THIS IS GOOD'
print(string.swapcase()) # >>> this is good

string = 'this is good'
print(string.swapcase()) # >>> THIS IS GOOD

string = 'ThiS iS gOOd'
print(string.swapcase())

'''
title()
    각 단어의 첫 글자를 대문자로 반환한다.
'''
string = 'this is good'
print(string.title()) # >>> This Is Good

'''
replace('바꿀문자열', '새문자열')
    : 특정 문자열을 찾아 변경한다. 
'''
#문자열에 직접 적용한다. 
print("="*5, "replace()", "="*20)
print('Hello, world!'.replace('world', 'Python'))

#변수에 적용한다. 
s = 'Hello, world!'
s = s.replace('world', 'Python')
print(s)

'''
문자바꾸기
    translate()는 문자열 내의 문자를 다른 문자로 변경한다. 
    maketrans('바꿀문자', '새문자')로 변환 테이블을 만든 후 적용한다. 
'''
print("="*5, "maketrans()/translate()", "="*20)
str1 = "apple"
'''
a e i o u
1 2 3 4 5 <= 이와 같은 변환테이블을 생성한다. 
즉 a->1, e->2와 같이 변환할 수 있게 매칭한다. 
'''
table = str1.maketrans('aeiou', '12345')
#앞에서 생성한 변환테이블을 적용하여 해당 문자를 변경한다.
print(str1.translate(table)) # 결과: 1ppl2

# '한'=>X, '아'=>Y 로 변경된다.
str2 = '한글은 아름다운 언어'
table = str2.maketrans('한아언', 'XYZ')
print(str2.translate(table))

'''
join(iterable)
    : join 은 반복 가능한 객체에 사이사이에 문자열을 연결하게 해준다.
    iterable : 리스트 ,튜플, 리스트, 문자열 등
'''
# 리스트 join
numList = ['1', '2', '3', '4']
separator = ','
print(separator.join(numList))

'''
'구분자'.join(합칠문자열)
    => 합칠 문자열을 구분자를 통해 합쳐서 반환한다. 
'''
print("="*5, "join()", "="*20)
print('-'.join(['010','1234','5678']))

'''
공백삭제 혹은 특정문자삭제
: lstrip(왼쪽) , rstrip(오른쪽), strip(양쪽)
별도의 인자가 없으면 공백이 삭제된다. 
'''
print("="*5, "strip()", "="*20)
str = "#^%오늘은 @@ 파이썬 @@ 공부하는날%^%"
print(str.lstrip('#')) #왼쪽의 # 제거
print(str.rstrip('%')) #오른쪽의 %제거

#좌우측의 특수기호와 문장에 포함된 @를 모두 제거한다. 
#즉, 메서드체인을 사용할 수 있다. 
print(str.rstrip('%').lstrip('#').replace('@', ''))

'''
문자열의 위치 찾기 인덱스
    : find(왼쪽부터찾기), rfind(오른쪽부터찾기)
'''
print("="*5, "find()", "="*20)
print('apple pineapple'.find('pl'))
print('apple pineapple'.rfind('pl'))