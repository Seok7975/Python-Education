'''
open()
: 파일을 다룰때 사용되는 내장함수로 첫번째 인자인 file경로만 
필수인자이다. 
형식]
    open(파일경로, mode='', buffering='', encoding='')    

mode : 파일열기모드
    r : 읽기모드
    w : 쓰기모드
    a : 추가모드. 파일의 마지막 부분에 새로운 내용을 추가한다.
    b : 2진모드(바이너리모드)로 컴퓨터가 알아볼 수 있는 파일을
        생성한다. 
    t : 텍스트모드로 디폴트 값이다. 사람이 알아볼 수 있는 형태의 
        텍스트파일을 생성한다. 메모장으로 열수있다. 
'''
print("="*30)
print("새파일01.txt")
print("="*30)

# 새로운 파일을 생성하여 반복문으로 내용을 입력한다. wt이므로 읽기/
# 텍스트 모드로 오픈한다. 
f_open = open("새파일01.txt", mode='wt', encoding='utf-8')
# 20번 반복하여 문자열을 입력한다. 
for i in range(1,21):
    # 서식문자를 이용해서 문자열을 구성한다. 
    data = "%d번째 줄입니다.\n" % i
    f_open.write(data)
f_open.close()

# 파일 읽기
f_read = open("새파일01.txt", mode='rt', encoding='utf-8')
while True:
    #파일 내용 한줄을 읽는다.
    line = f_read.readline()
    #더 이상 읽을 내용이 없으면 반복문을 탈출한다.
    if not line: break
    #출력
    print(line)
f_read.close()

# 기존 파일에 내용 추가하기
f_add = open("새파일01.txt", mode='at', encoding='utf-8')
# 한줄을 추가한다. 개행문자가 없으므로 줄바꿈 처리는 되지않는다.
f_add.write("추가하는 내용입니다.")
# 리스트를 통해 여러줄의 내용을 입력한다. 
f_add.writelines(["줄바꿈은 되나요?\n", "안되면 개행문자를 넣어주세요."])
f_add.write("마지막 라인입니다.")
f_add.close()

print("="*30)
print("새파일02.txt")
print("="*30)
# 자동으로 파일객체 닫기 및 여러줄 입력하기
# 파일에 내용 입력하기 
with open("새파일02.txt", mode='wt', encoding='utf-8') as myfile:
    for i in range(1, 16):
        data = "%d 라인 입력합니다.\n" % i
        myfile.write(data)

# 파일의 내용 읽기
'''
readline() 으로 파일을 읽을 때는 while 반복문을 활용한다.
더 이상 읽을 줄이 없을 때는 빈 문자열을 반환하는데, 이를 
조건으로 반복문을 탈출한다. 
이와같은 방법으로 길이에 상관없이 파일을 읽어올 수 있다. 
'''
with open("새파일02.txt", mode='rt', encoding='utf-8') as myfile:
    line = None
    while line != '':
        line = myfile.readline()
        print(line.strip('\n'))

'''
피클링(Pickling)
    : 파이썬 객체를 파일에 저장하는 과정을 의미한다. 복원하는것을
    언피클링이라고 한다. 
'''
# 피클링을 위한 모듈 임포트
import pickle

#파일에 저장할 내용준비. 문자열, 숫자, 딕셔너리..
name = '개발자'
age = 99
address = '서울시 구로구 디지털로'
times = {'java': 20, 'HTML':2, 'Oracle':10, 'Python':3}

# 쓰기모드로 바이너리 파일을 오픈한다. 만약 파일이 없다면 생성된다. 
with open('developer.p', 'wb') as file:
    # dump()함수를 통해 데이터를 저장한다. 
    pickle.dump(name, file)
    pickle.dump(age, file)
    pickle.dump(address, file)
    pickle.dump(times, file)
# 바이너리 파일을 읽기모드로 오픈한 후 load()함수를 통해 복원한다. 
with open('developer.p', 'rb') as file:
    name = pickle.load(file)
    age = pickle.load(file)
    address = pickle.load(file)
    times = pickle.load(file)
    # 복원한 내용을 출력한다. 
    print("이름", name)
    print("나이", age)
    print("주소", address)
    print("배당시간", times)