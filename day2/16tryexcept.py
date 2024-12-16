'''
예외처리
    : 에러를 핸들링하기 위해 사용된다. 
    파이썬에서는 try ~ except를 사용한다. 그리고 else를 사용할수있다. 
형식]
    try:
        예외발생 가능성이 있는 실행구문
    except:
        예외가 발생하면 처리
    else:
        예외가 발생하지 않으면 실행되는 구문
    finally:
        예외발생과 상관없이 try문으로 진입하면 항상 실행되는 구문
''' 
# 함수선언
def calc(val):
    sum=None
    #예외발생이 예상되는 지점을 try로 묶어준다. 
    try:
        sum = val[0] + val[1] + val[2]
        
        if val[0] == 100:
            # 선언하지 않는 변수를 출력하므로 예외발생
            print(no_var)
        elif val[0]==55:
            # 숫자를 0으로 나누면 무한대가 되므로 예외발생
            result = val[0] / 0
            print("결과", result)
    #예외명을 명시하여 특정 예외만 처리한다.
    except IndexError:
        print('리스트의 인덱스에 에러가 있습니다.')
    #예외발생시의 에러메세지를 변수로 받아서 출력할때는 as를 사용한다.
    except NameError as err:
        print('선언되지 않은 변수를 사용하였습니다.', str(err))
    #예외명을 명시하지 않으면 모든 예외를 처리할 수 있는 블럭이된다.
    except:
        print('예외가 발생하였습니다.')
    #예외가 발생하지 않았을때 실행되는 블럭
    else:
        print('에러 없음~~')
    #try로 진입했을때 예외발생과 상관없이 무조건 실행되는 블럭
    finally:
        print('sum', sum)
#리스트의 원소가 3개이므로 정상적으로 실행된다.
print('실행1') 
calc([1,2,3])

#원소가 2개밖에 없으므로 "IndexError"가 발생된다. 
print('실행2')
calc([10,20])

#no_var라는 변수를 선언한적이 없으므로 "NameError"가 발생된다. 
print('실행3')
calc([100,101,102.103])

#0으로 나누기 때문에 계산불능 예외가 발생된다. 
print('실행4')
calc([55,56,57])

print('실행5')
try:
    #텍스트 파일을 읽기전용으로 오픈한다.
    fp = open("text.txt", "rt")
    # fp = open("새파일01.txt", mode="rt", encoding='utf-8')
    try:
        #내용을 한줄 읽어온다. 
        lines = fp.readlines()
        print(lines)
    finally:
        #예외와 상관없이 파일 객체를 닫아준다.
        print("파일객체를 닫습니다.")
        fp.close()
except IOError:
    #현재 경로에는 해당 파일이 없으므로 예외가 발생한다.
    print('파일 에러 발생')

#개발자가 직접 예외 발생시키기
print('실행 6')
try:
    x = int(input('3의 배수를 입력하세요: '))
    #입력한 수가 3의 배수가 아니면 예외를 발생시킨다. 
    if x % 3 != 0:
        '''예외를 발생시킬때 Java와 같이 throws를 사용하지 않고
        raise를 사용한다. '''
        raise Exception('[예외 메세지] 3의 배수가 아님')
    print(x)
except Exception as e:
    print("예외가 발생했습니다.", e)
    
'''
개발자 정의 예외클래스 생성
: 예외 클래스 생성시 Exception 객체를 상속하고 생성자에서 
예외발생시 출력할 메세지를 정의한다. 
''' 
print('실행 7')
#파이썬에서는 extends를 사용하지않고, 매개변수와 같은 형태로 상속한다.
class GugudanRangeExcept(Exception):
    #생성자 메서드
    def __init__(self):
        super().__init__('구구단의 범위를 벗어났습니다.')
def print_gugudan(end_num):
    try:
        '''단수가 9를 초과하는 경우 예외객체를 생성하여 raise한다.
        '''
        if end_num > 9 : 
            raise GugudanRangeExcept
        end_num = end_num + 1
        for dan in range(2, end_num):
            for su in range(1, 10):
                print("%2d * %2d = %2d" % (dan, su, dan*su), end=' ')
            print()
        print()
    except Exception as e:
        '''예외객체 생성시 생성자를 통해 지정했던 예외메세지가
        출력된다. '''
        print('예외발생', e)

print_gugudan(int(input("출력할 단수를 입력하세요:")))