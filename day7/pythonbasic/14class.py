'''
인스턴스 변수
    : 메서드 안에 정의되는 변수
    -클래스 내부에서는 'self.변수명'으로 접근한다.
    -클래스 외부에서는 '객체변수.인스턴스변수'로 접근한다 
인스턴스 메서드
    : 인스턴스 변수에 항상 접근할 수 있도록 메서드의 첫번째 
    매개변수에 항상 객체 자신을 의미하는 self를 선언해야한다. 
    self인수를 생략하면 오류가 발생한다.
'''
# 클래스 선언
class FourCaluculator:
    #setter메서드로 초기화를 담당한다. 
    def setdata(self, first, second):
        # self : 멤버변수에 값을 초기화 하거나 메서드 안에서 멤버변수를 
        #       참조하거나 또는 다른 메서드를 호출할 경우 이용한다.
        #self.멤버변수 = 매개변수 형태로 기술하여 초기화한다. 
        self.first = first
        self.second = second
    '''
    사칙연산을 수행하는 인스턴스 메서드를 정의함. 멤버변수에 접근하기
    위해 첫번째 파라미터로 self를 반드시 사용해야한다. 
    '''
    def addition(self):
        result = self.first +  self.second
        return result
    def subtraction(self):
        result = self.first -  self.second
        return result
    def mutiplication(self):
        result = self.first *  self.second
        return result
    def division(self):
        result = self.first /  self.second
        return result
''' 객체생성. 파이썬에서는 new를 사용하지 않는다. 최근에 나온
언어들은 대부분 new없이 객체를 생성한다. (코틀린, 다트 등) '''
a = FourCaluculator()
b = FourCaluculator()

'''객채 생성 후 setter()를 통해 인스턴스 변수를 초기화한다. 
self는 함수 생성시 반드시 필요한 것이므로 호출시에는 무시해야한다.
매개변수가 아니므로 기술하면 에러가 발생한다. '''
a.setdata(4, 2)
b.setdata(3, 8)
    
# 각 객체의 인스턴스 메서드를 호출한다. 
print("객체a 덧셈", a.addition())
print("객체a 곱셈", a.mutiplication())
print("객체b 뺄셈", b.subtraction())
print("객체b 나눗셈", b.division())
print("="*40)

'''
생성자(Initializer)
    : 클래스로부터 객체를 생성할때 인스턴스 변수를 초기화한다. 
    init양쪽에 언더바 2개를 붙여 정의한다. 
정적메서드
    : 클래스명으로 바로 접근할 수 있는 메서드를 말한다. 
    @staticmethod라는 데코레이터를 사용한다. 
클래스메서드
    : 정적메서드와 비슷한데, 객체 인스턴스를 의미하는 self대신
    cls라는 클래스를 의미하는 파라미터를 전달받는다. 
    cls를 통해 클래스변수에 엑세스 할 수 있다. 
    @classmethod 라는 데코레이터를 사용한다. 
클래스변수
    : 클래스 전체에서 접근가능한 전역변수를 말한다.
'''
class CalculatorInit:
    # 클래스변수 생성
    count = 0
    '''
    def __init__(self):
    파이썬에서는 Java와 같이 생성자 오버로딩은 지원하지 않는다. 
    
    생성자 메서드 정의. 멤버변수 초기화를 위해 첫번째 파라미터는
    self를 필수로 기술해야한다. 
    '''
    def __init__(self, first, second):
        #인스턴스 변수 생성
        self.first = first
        self.second = second
        #생성자에서 클래스변수 접근 가능
        CalculatorInit.count += 1
    
    #일반적인 인스턴스 메서드 정의
    def addition(self):
        result = self.first + self.second
        return result
    
    '''
    @staticmethod 데코레이터를 통해 정적메서드로 정의한다. 
    정적메서드는 객체생성없이 클래스명으로 직접 호출할 수 있다. 
    '''
    @staticmethod
    def staticArea(pFirst, pSecond):
        result = pFirst + pSecond
        print("static 메소드", result)
    
    '''클래스메서드 정의. cls를 통해 클래스변수에 접근할 수 있다. '''
    @classmethod
    def showInfo(cls):
        print('class 메소드', cls.count)
'''
fCal = CalculatorInit() 
인수가 없는 기본생성자는 정의되지 않았으므로 이와 같이 기술하면
실행시 에러가 발생한다. 
'''
# fCal = CalculatorInit()
# 2개의 인수를 통해 객체를 생성한다. 
fCal = CalculatorInit(2010, 43)
# 일반적인 인스턴스 메서드 호출
print(fCal.addition())
# 클래스 메서드 호출1 : 인스턴스 변수를 통해 호출한다.(권장사항X)
fCal.showInfo()
# 클래스 메서드 호출2 : 클래스명을 통해 호출한다.(권장사항임)
CalculatorInit.showInfo()
# 정적메서드 호출 : 클래스명을 통해 직접 호출한다. 
CalculatorInit.staticArea(5,8)

'''
상속 : 파이썬에서는 별도의 키워드없이 클래스 정의시 매개변수 
형태로 상속한다. 
'''
class MoreCalculator(CalculatorInit):
    # 자식쪽에서 확장한 인스턴스 함수
    def pow(self):
        # first를 second만큼 거듭제곱하여 반환한다.
        result = self.first ** self.second
        return result
    # 오버라이딩 : 부모의 기능을 자식쪽에서 재정의한 함수. 
    def addition(self):
        return (self.first + self.second) * 2
#객체생성 및 함수 호출
moreCal = MoreCalculator(4, 3)
# 4의 3승의 결과를 출력한다.
print("성속 후", moreCal.pow())

#부모클래스와 자식클래스를 통해 각각 객체를 생성한다. 
p1 = CalculatorInit(100, 200)
p2 = MoreCalculator(100, 200)
#부모의 메서드는 단순히 합의 결과를 반환한다.
print("부모객체로 호출",p1.addition())  #결과 : 300
#자식의 메서드는 합의 결과에 2를 곱해 반환한다. 
print("자식객체로 호출", p2.addition()) #결과 : 600
#즉, 오버라이딩 된 메서드이므로 호출한 객체에 따라 다른 결과가 출력된다. 
print("="*40)

#정보은닉(캡슐화) : 인스턴스 변수를 비공개속성(private)으로 설정할 수 있다.
class Person:
    #생성자 메서드 정의
    def __init__(self, n, a, pw):
        #언더바가 없으면 public변수가 된다.
        self.name = n
        self.age = a
        #언더바를 앞에 2개를 붙여주면 private변수가 된다. 
        self.__passwd = pw
    # getter 역할의 메서드로 private멤버변수를 반환한다. 
    def secret_info(self):
        return self.__passwd

# 3개의 인수를 통해 객체를 생성한다. 
p1 = Person('my', 22, 'qwer123')
# public 멤버는 직접 접근해서 출력할 수 있다.
print("이름", p1.name)
print("나이", p1.age)
# private 멤버는 클래스외부에서는 접근할 수 없으므로 에러가 발생한다.
# print("비밀번호1", p1.__passwd) # 에러발생
# print("비밀번호2", p1.passwd) # 이런 변수가 없으므로 에러발생
# private 변수는 메서드를 통해 간접적으로 접근하여 호출해야한다.  
print("비밀번호3", p1.secret_info()) 
