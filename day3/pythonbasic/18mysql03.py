import pymysql

#DB 연결
conn = pymysql.connect(host='localhost', user='sample_user',
                       password='1234', db='sample_db', charset='utf8')
#커서생성
curs = conn.cursor()

#쿼리 작성시 format() 함수를 통해 문자열을 인덱스로 매칭한다.
sql = """UPDATE board
            SET title='{1}', content='{2}'
            WHERE num={0}
      """.format(input('수정할일련번호:'), input('제목'), input('내용'))
try:    
    curs.execute(sql)
    conn.commit()
    print("1개의 레코드가 수정됨")
except Exception as e:
    #쿼리실행시 문제가 있는 경우 rollback 처리한다. 
    conn.rollback()
    print("쿼리 실행시 오류발생", e)

conn.close()