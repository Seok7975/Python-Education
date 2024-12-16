#곱셈, 나눗셈을 위한 함수 선언
def mul(a, b):
    return a * b
def div(a, b):
    return a / b

'''
__name__ 변수란? => 시작점을 구별할 수 있는 기준
    : 파이썬의 __name__ 변수는 내부적으로 사용하는 특별한 변수명이다. 
    만약 콘솔에서 직접 mod2.py를 실행할 경우에는 해당 변수에 __main__
    이 저장된다. 하지만 import를 하는 경우 모듈이름인 mod2가 저장된다.
    즉, 해당 모듈이 직접 실행되는지 임포트되어 실행되는지를 구분하기
    위한 변수이다. 
    호출하면 __main__ & 호출당하면 모듈
'''
if __name__ == "__main__":
    print("==여긴 mod2.py==")
    print(mul(1, 4))
    print(div(4, 2))