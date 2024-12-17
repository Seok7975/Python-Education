import pymysql

#DB 연결
conn = pymysql.connect(host='localhost', user='sample_user',
                       password='1234', db='sample_db', charset='utf8')
#커서생성
curs = conn.cursor()

#f-String으로 delete쿼리문 작성
sql = f"""DELETE FROM board WHERE num='{input('삭제할 일련번호:')}'"""

try:    
    curs.execute(sql)
    conn.commit()
    print("1개의 레코드가 삭제됨")
except Exception as e:
    #쿼리실행시 문제가 있는 경우 rollback 처리한다. 
    conn.rollback()
    print("쿼리 실행시 오류발생", e)

conn.close()