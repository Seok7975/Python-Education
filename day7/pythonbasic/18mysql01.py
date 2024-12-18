'''
파이썬에서는 MySQL(MariaDB) 사용을 위해 PyMySQL 모듈을 설치해야
한다. 
c:> pip3 install pymysql
''' 
# MySQL사용을 위한 모듈 임포트
import pymysql

# MySQL 연결
conn = pymysql.connect(host='localhost', user='sample_user',
                       password='1234', db='sample_db', charset='utf8')
'''
    , cursorclass=pymysql.cursors.DictCursor
    => 위 설정이 없는 경우 레코드를 튜플로 출력한다. 
    해당 설정을 통해 딕셔너리로 출력할 수 있다. 
'''
# 커서 생성
curs = conn.cursor()

try:
    #board테이블의 전체 레코드를 select한다.
    sql = "SELECT * FROM board"
    curs.execute(sql)
    
    #select한 모든 레코드 인출. 반복문 없이 전체를 출력한다. 
    rows = curs.fetchall()
    print(rows)
    
    # 행(레코드) 단위로 출력한다.
    print('출력1', '-'*40)
    for row in rows:
        print(row)
        
    # 행 단위로 출력하되 각 컬럼의 인덱스를 지정하여 출력한다.
    print('출력2', '-'*40)
    for row in rows:
        print(row[0], row[1], row[2], end=" ")
        pdate = row[3]
        id = row[4]
        vcnt = row[5]
        print("%s, %s, %s" % (pdate, id, vcnt))
    
    #format 함수를 통해 쿼리문에 사용자가 입력한 검색어를 추가한다.
    print('출력3', '-'*40)
    sql = sql + " WHERE title like '%{0}' ".format(input('검색어입력:'))
    curs.execute(sql)
    rows = curs.fetchall()
    print(rows)
except Exception as e:
    print("쿼리 실행시 오류발생", e)

print('-'*40)
conn.close()
print('자원반납')
