'''
웹크롤링을 위해 두가지 모듈(라이브러리) 설치가 필요하다. 
pip3 install requests
pip3 install beautifulsoup4
'''
# 모듈 임포트
import requests
from bs4 import BeautifulSoup

# 네이버 지식인에서 '파이썬'으로 검색한 후 URL 복사
url = 'https://kin.naver.com/search/list.naver?query=%ED%8C%8C%EC%9D%B4%EC%8D%AC'
# 접속을 위해 requests 모듈을 사용한다.
response = requests.get(url)

# 응답코드가 200이면 접속성공이므로 크롤링을 시작한다.
if response.status_code == 200:
    # HTML코드 전체를 받아온다.
    html = response.text
    # 파싱하기 위해 soup객체를 생성한다. 이때 HTML파서를 사용한다.
    soup = BeautifulSoup(html, 'html.parser')
    
    # CSS셀렉터를 통해 원하는 엘리먼트를 추출한다.
    title1_1 = soup.select_one('#s_content > div.section > ul > li:nth-child(1) > dl > dt')
    # print("추출1_1:", title1_1)
    
    #파이어폭스에서의 셀렉터를 통해 추출
    # title1_2 = soup.select_one('.basic1 > li:nth-child(1) > dl:nth-child(2) > dt:nth-child(1)')
    # print("추출1_2:", title1_2)
    
    text = title1_1.get_text()
    # print("추출2:", text)
    
    # 검색결과 부분에서 <li>태그로 반복되는 부분을 크롤링한다.
    ul = soup.select_one('ul.basic1')
    # print("추출3:", ul)
    
    #지식인에서 검색된 목록중에서 제목만 텍스트로 뽑아서 출력한다.
    print("추출4")
    titles2 = ul.select('li>dl>dt>a')
    for tit in titles2:
        print(tit.get_text())
else:
    print(response.status_code) 