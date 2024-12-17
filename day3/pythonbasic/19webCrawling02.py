# 모듈 임포트
import requests
from bs4 import BeautifulSoup

response = requests.get('https://www.koreabaseball.com/Record/Player/HitterBasic/BasicOld.aspx?sort=HRA_RT')
html = response.text
soup = BeautifulSoup(html, 'html.parser')
# print(soup)

#타이틀 엘리먼트 가져오기
title = soup.select_one('#contents > h4')
# print("title 요소: ",title)

#타이틀 텍스트만 추출하기
title_text = title.get_text()
print("title 텍스트: ", title_text)

#타자기록 테이블 가져오기
record_table = soup.select_one('#cphContents_cphContents_cphContents_udpContent > div.record_result > table')
# print("타자기록 요소 :", record_tr)

# <tr>태그를 가져온 후 갯수만큼 반복하여 파싱한다.
record_tr = soup.select_one('#cphContents_cphContents_cphContents_udpContent > div.record_result > table > tbody')
repeat_tr = record_tr.select('tr')
for rec in repeat_tr:
    # print("ddd", rec)
    d1 = rec.select_one('td:nth-child(1)').get_text() # 순위
    d2 = rec.select_one('td:nth-child(2)').get_text() # 선수명
    d3 = rec.select_one('td:nth-child(3)').get_text() # 팀명
    d4 = rec.select_one('td:nth-child(4)').get_text() # 타율
    d5 = rec.select_one('td:nth-child(5)').get_text() # 경기
    d6 = rec.select_one('td:nth-child(6)').get_text() # 타석
    d7 = rec.select_one('td:nth-child(7)').get_text() # 타수
    d8 = rec.select_one('td:nth-child(8)').get_text() # 안타
    d9 = rec.select_one('td:nth-child(9)').get_text() # 2루타
    d10 = rec.select_one('td:nth-child(10)').get_text() # 3루타
    d11 = rec.select_one('td:nth-child(11)').get_text()  # 홈런
    d12 = rec.select_one('td:nth-child(12)').get_text() # 타점
    d13 = rec.select_one('td:nth-child(13)').get_text() # 도루
    d14 = rec.select_one('td:nth-child(14)').get_text() # 도루실패
    d15 = rec.select_one('td:nth-child(15)').get_text() # 볼넷
    d16 = rec.select_one('td:nth-child(16)').get_text()  # 사구
    d17 = rec.select_one('td:nth-child(17)').get_text() # 삼진
    d18 = rec.select_one('td:nth-child(18)').get_text() # 병살타
    d19 = rec.select_one('td:nth-child(19)').get_text() # 실책
    #출력
    print(d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19)
    