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
for i, v in enumerate(data):
    print(i, v)