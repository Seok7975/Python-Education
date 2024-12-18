# sqlite를 사용하기 위한 모듈 임포트
import sqlite3
# 파일형태로 존재하는 dbase1에 연결한다. 최초 실행이라면 파일이
# 생성된다. 
conn = sqlite3.connect('dbase1.db')
# 연결후 DB작업을 위해 커서를 생성한다. 
curs = conn.cursor()

'''첫 실행시에는 문제가 없으나 두번째 실행에서는 이미 테이블이
존재하므로 예외처리를 해야 아래 문장이 실행된다. 
첫 실행에서 테이블이 생성된다. '''
try:
    tblcmd = 'create table people (name char, job char, pay int)'
    curs.execute(tblcmd)
except Exception as e:
    print("[예외발생] 테이블은 이미 생성 되었습니다.", e)
    
#레코드 삽입
#방법1 : 한개의 레코드를 삽입한다. 튜플을 이용해서 인파라미터를 설정한다.
# curs.execute('insert into people values (?,?,?)', ('이순신','장군',500))

#방법2 : 2개 이상의 레코드를 삽입할때는 리스트를 활용한다. 
# curs.executemany("insert into people values (?,?,?)", 
# 		[('강감찬','장군',700), ('홍길동','의적',800)])

#방법3 : for문을 이용해서 반복적으로 삽입한다. 
# rows = [['곽재우', '의병',1100],['유성룡','재상',1200],['임꺽정','의적',1500]]
# for row in rows:
#     curs.execute('insert into people values (?,?,?)', row)

#커밋 
# conn.commit()

#조회1 : 조회된 레코드를 한꺼번에 출력한다. 
# curs.execute("SELECT * FROM people")
# print(curs.fetchall())
# print('---------------------------------')

#조회2 : 조회된 결과를 한 행(row)씩 출력한다. 
# curs.execute("SELECT * FROM people")
# for row in curs.fetchall():
#     print(row)
# print('---------------------------------')

#조회3 : 각 컬럼별로 출력한다. 
# curs.execute("SELECT * FROM people")
# for (name, job, pay) in curs.fetchall():
#     print(name, ':', job, ':', pay)
    
# '''
# 연습문제] 이름이 '강감찬'인 레코드의 급여를 9999로 변경하시오.
# '''
#curs.execute("update people set pay=값 where name=값")
# curs.execute("update people set pay=? where name=?",
#              (9999, '강감찬'))

# '''
# 연습문제] 급여가 1200인 레코드를 삭제하시오.
# '''   
curs.execute("delete from people where pay=?",
                    (1200,))
# 변경된 레코드가 있으면 반드시 커밋해서 테이블에 적용한다. 
conn.commit() 