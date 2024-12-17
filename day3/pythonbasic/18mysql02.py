import pymysql

#DB 연결
conn = pymysql.connect(host='localhost', user='sample_user',
                       password='1234', db='sample_db', charset='utf8')
#커서생성
curs = conn.cursor()

'''f-String을 통해 쿼리문 작성. 변수를 문자열에 즉시 적용할 수 
있어 편리하다. input() 함수를 통해 사용자가 입력한 내용을 
문자열에 즉시 반영한다. '''
sql = f"""INSERT INTO board (title, content, id)
	VALUES ('{input('제목:')}', '{input('내용:')}', 'korea')"""
try:    
    #쿼리문 실행
    curs.execute(sql)
    #insert와 같이 행의 변화가 있는 쿼리문은 commit() 함수를 호출한다.
    conn.commit()
    print("1개의 레코드가 입력됨")
except Exception as e:
    #쿼리실행시 문제가 있는 경우 rollback 처리한다. 
    conn.rollback()
    print("쿼리 실행시 오류발생", e)

conn.close()