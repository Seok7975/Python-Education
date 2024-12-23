# -*- coding: utf-8 -*-
"""영화 추천 시스템
    https://colab.research.google.com/drive/1GecKL32OyM1Z-QbWdYxMbV5C0bog7_MA

TMDB5000 은 IMDB라고 하는 아주 큰 영화 관련 사이트에 있는 5000개의 유명한 영화 정보를 가공해서 케글에 제공하는 데이터 세트 입니다.

- Reference : https://www.kaggle.com/code/ibtesama/getting-started-with-a-movie-recommendation-system

kaggle은 전 세게적으로 수많은 AI 개발자 분들이 모여서 경진을 펼친는곳인데 여기에서는 다양한 데이터가 있고 그 데이터를 가지고 만든 예측 모델들을 자세한 셜명과 함께 올려 서로 의견을 주고 받으면서 공부를 할수 있는 곳입니다.

# 영화 추천 시스템

1. Demographic Filtering (인구통계학적 필터링)
- 많은 사람들이 일반적으로 좋아하는 아이템을 추첮하는 방식입니다. 예를 들어 천만관객 돌파한 영와등. 누가봐도 좋아할만한 아이템을 추천하는 것.

2. Content Based Filtering (컨텐츠 기반 필터링)
- 특정 아이템에 기반한 유ㅜ사 아이템을 추천하는 방식입니다. 예를들어 무슨 영화를 봒다 그러면 그 영화와 비슷한 장르나 감독, 줄거리나 장소 , 시간적 배경 또는 주연 배우들 등등의 요소가 고려해서 비슷한 영화를 좋아할 거라고 예상하고 추천해주는 것.

3. Collaborative Filtering (협업 필터링)
- 비슷한 영화 취향을 가진 사람들끼리를 매칭시켜서 추천해주는 방식. 내가 지금까지 평가한 이력 또는 물건을 구매한 이력을 기반으로 추천해주는 방식.

## 1. Demographic Filtering (인구통계학적 필터링)
"""

import pandas as pd
import numpy as np

df1 = pd.read_csv(',/data/tmdb_5000_credits.csv')
df2 = pd.read_csv('./data/tmdb_5000_movies.csv')

df1.head()

"""movie_id	고유 아이디, title	영화 제목, cast 영화배우 정보, crew 영화 감독이나 작가등의 정보"""

df2.head(3) # 컬럼이 많아 3개만 불러옴.

"""budget 영화 예산, genres 영화 장느, id 위의 데이터의 movie_id와 같음. 	popularity 인기도, vote_average 영화평점,	vote_count 영화 평수"""

df1.shape, df2.shape # 두개의 테이블을 합쳐서 사용하겠음.

df1['title'].equals(df2['title']) # 두 테이블을 합치기 위해 타이틀이 같은지 비교

df1.columns # df1의 컬럼 확인하기

df1.columns = ['id', 'title', 'cast', 'crew'] # movie_id를 id로 변경
df1.columns

df1[['id', 'cast', 'crew']]  # title'은 같기 때문에 삭제함.

df2 = df2.merge(df1[['id', 'cast', 'crew']], on='id') # 두데이터를 on='id'를 기준으로 합침.(merge)
df2.head(3)

"""대중들이 좋아하는 영화를 찾기 위해 신뢰도가 높은 영화를 찾기 위해 가중치 적용.

영화 1 : 영화의 평점이 10/10 -> 5명이 평가

영화 2 : 영화의 평점이 8/10 -> 500명이 평가 => 신뢰도가 높음.

영화의 가중치 계산

- v : 영화의 총 평가 수. vote_count
- m : 차트에 포함되기 위한 최소한의 평가
- R : 영화의 평균 평점. vote_average
- C : 전체 평점의 평균 정보
"""

C = df2['vote_average'].mean() # 전체 평점의 평균 정보
C

m = df2['vote_count'].quantile(0.9) # 평가 순위가 많은 상위 10%(quantile(0.9)) 에 대한 데이터를 뽑아 준다.
m # 상위 10%에 해당하는 평가수의 기준이 1938개 됩니다. 즉 이것보다는 많은 평가가 있는 영화를 고르면 됩니다.

q_movies = df2.copy().loc[df2['vote_count'] >= m] # 1838 보다 평가가 크거나 같은 데이터만 가져와서 df2에 복사 q_movies에 넣어줌.
q_movies.shape

q_movies['vote_count'].sort_values() # vote_count 수로 정렬 가장 적은 개 1840개임.

# 가중치 계산. 그림에 있는 계산식을 함수 로 만듬.
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C) # 그림에 있는 공식

q_movies['score'] = q_movies.apply(weighted_rating, axis=1) # 가중치 계산으로 산출된 값이 스코어의 값으로 들어감. axis=1 열(컬럼) 단위로 전달.
q_movies.head(3) # 끝에 score가 추가된 것을 볼수 있음.

q_movies = q_movies.sort_values('score', ascending=False) # 스코어 점수로 내림차순(ascending=False)으로 정렬
q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10) # 최대 10개까지만 대중적으로 인기있는

# 데이터에 있는 popularity(인기가 많은 순으로) 기준으로 정렬 및 시각화
pop= df2.sort_values('popularity', ascending=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))

plt.barh(pop['title'].head(10),pop['popularity'].head(10), align='center',
        color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")

"""우리가 산정한 스코어와 여기서 제공하는 인기순(popularity 여러가지 상황으로 만든 인기순클릭수와 검색등등)과의 결과가 다름.

## 2. Content Based Filtering (컨텐츠 기반 필터링)

### 줄거리 기반 추천
"""

df2['overview'].head(5) # 영화에대한 줄거리(텍스트를 분석하여 )

"""#### 텍스트 분석

Bag Of Words - BOW : 텍스트에 포함된 모든 단어들이 순서 상관없이 각각 몇 번씩 나왔는지 개수를 보는 것.

문장1 : I am a boy

문장2 : I am a girl

I(2), am(2), a(2), boy(1), girl(1) # () 안에는 나온 개수

        
           I    am   a   boy    girl
    문장1  1    1    1    1      0   (1,1,1,1,0)
    (I am a boy)
    문장2  1    1    1    0      1   (1,1,1,0,1)
    (I am a girl)



 피처 벡터화.: 각 문장별로 위 단어들이 나온 횟수를 이렇게 매트릭스로 만드는 것.


 문서 100개
 모든 문서에서 나온 단어 10,000 개<br/>
 100 * 10,000 = 100만

           단어1, 단어2, 단어3, 단어4, .... 단어 10000
    문서1    1       1       3    0    
    문서2
    문서3
    ..
    문서100



### 피처 벡터화 유형(자연어 처리)
1. TfidfVectorizer (TF-IDF 기반의 벡터화) : TF-IDF는 단어 빈도(TF)와 문서 빈도의 역수(IDF)를 곱한 값입니다. 중요도의 차이를 가지고 계산 즉 관사(THE, A)같은 것은 중요도가 떨어짐.
2. CountVectorizer : 각 단어가 문서에서 몇 번 출현했는지를 기반으로 문서를 벡터화
"""

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english') # 영어에서 크게 의미 없는 단어들을 제외해 주는 것. the 나  a 같은...

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS # 영어에서 크게 의미 없는 단어
ENGLISH_STOP_WORDS # 이런단어들을 피처링해서 제외(무시) 시킨다.

df2['overview'].isnull().values.any() # null 이 있는지 확인 True이므로 있다는 것임.

df2['overview'] = df2['overview'].fillna('') # null 값이 있으므로 이 null 값에 빈값으로 채워 준다. 빈값으로 채워준것을 다시 overview에 넣어준다.

tfidf_matrix = tfidf.fit_transform(df2['overview'])
tfidf_matrix.shape # 4803개의 오버뷰의 문서들이 20978개의 단어들로 이루어졌다는 말

tfidf_matrix # 125840 개의 00이아닌 데이터가 있다는 것.

from sklearn.metrics.pairwise import linear_kernel # linear_kernel 함수가 코사인 유사도와 비슷하지만 더 빠름

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim

"""| | 문장1 | 문장2 | 문장3 |
|---|---|---|---|
|문장1|1|0.3|0.8|
|문장2|0.3|1|0.5|
|문장3|0.8|0.5|1|

1인 자신외의 코사인 유사도가 가장 높은 다른 문장을 찾는 것. 예)문장 1일때는 0.8인 문장 3 대칭인 것을 알수 있다.
"""

cosine_sim.shape

"""영화를 추천하기 위한 함수를 만들 것임."""

# 함수에 필요한 데이터 처리(영화제목으로 그영화가 전체 데이터중 몇번째 해당하는 제목인지 알기 위해서 인덱스 값 설정)
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates() # 영화제목를 구하기 위한 인덱스 값 설정 . drop_duplicates() : 중복 제거
indices # 우측에있는 값이 인덱스 값

indices['The Dark Knight Rises'] # 이제목의 인덱스 값은 3 이렇게 만드 인덱스 값으로 제목을 얻어 올수 있다.

df2.iloc[[3]] # 인덱스 3을 넣으면 The Dark Knight Rises 정보가 나옴.

"""##### 영화 추천 함수 만들기"""

# 영화의 제목을 입력받으면 코사인 유사도를 통해서 가장 유사도가 높은 상위 10개의 영화 목록 반환
def get_recommendations(title, cosine_sim=cosine_sim):  # (영화제목, 코사인유사도)
    # 영화 제목을 통해서 전체 데이터 기준 그 영화의 index 값을 얻기 - 1
    idx = indices[title]

    # 코사인 유사도 매트릭스 (cosine_sim) 에서 idx 에 해당하는 데이터를 (idx, 유사도) 형태로 얻기 - 2
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 코사인 유사도 기준으로 내림차순 정렬 - 3
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 자기 자신을 제외한 10개의 추천 영화를 슬라이싱 - 4
    sim_scores = sim_scores[1:11]

    # 추천 영화 목록 10개의 인덱스 정보 추출 - 5
    movie_indices = [i[0] for i in sim_scores]  # 위에 슬라이싱을 했기 때문에 않해도 됨.

    # 인덱스 정보를 통해 영화 제목 추출 - 6
    return df2['title'].iloc[movie_indices]

test_idx = indices['The Dark Knight Rises'] # 영화 제목을 통해서 전체 데이터 기준 그 영화의 index 값을 얻기 - 1 이것 먼저 하고 위에 것을 할 것.
test_idx

cosine_sim[3]

test_sim_scores = list(enumerate(cosine_sim[3])) # 코사인 유사도 매트릭스 (cosine_sim) 에서 idx 에 해당하는 데이터를 (idx, 유사도) 형태로 얻기 - 2

# sorted : 새로운 데이터를 정렬
test_sim_scores = sorted(test_sim_scores, key=lambda x: x[1], reverse=True) # test_sim_scores에서 코사인 유사도 기준으로 내림차순 정렬 - 3. key=lambda x: x[1] : 1번 인덱스를 기준으로 정렬했다는 뜻.

test_sim_scores[1:11] # 자기 자신을 제외한 10개의 추천 영화를 슬라이싱 - 4

"""### 람다식

아주 작은 함수
"""

def get_second(x):  # 일반식
    return x[1] # 인덱스 1의 값을 출력

lst = ['인덱스', '유사도']
print(get_second(lst))

(lambda x: x[1])(lst) # 위에 식을 람다 식으로 표현

# 추천 영화 목록 10개의 인덱스 정보 추출 - 5
test_movie_indices = [i[0] for i in test_sim_scores[1:11]]
test_movie_indices

# 인덱스 정보를 통해 영화 제목 추출  - 6
df2['title'].iloc[test_movie_indices]

df2['title'][:20] # 20개의 타이틀 컬럼을 호출

get_recommendations('Avengers: Age of Ultron')  # 함수를 사용해 Avengers: Age of Ultron를 넣었을 때 추천하는 영화

get_recommendations('The Avengers')

"""### 다양한 요소 기반 추천 (장르, 감독, 키워드 등)


"""

df2.head(3)

df2.loc[0, 'genres'] # 0번째 로우에 해당하는 장르만 가지고 실행

s1 = [{"id": 28, "name": "Action"}]
s2 = '[{"id": 28, "name": "Action"}]'

type(s1), type(s2)

from ast import literal_eval  # literal_eval : 문자열로 되어있는 것을 리스트 형식으로 바꾸는 작업 s2를 s1으로 바꾸는 것.
s2 = literal_eval(s2)
s2, type(s2)

print(s1)
print(s2)

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(literal_eval) # features 에 있는 위의 4개의 컬럼에 있는 모든 데이터를 문자열에서 리스트형으로 바꿔준는 것.

df2.loc[0, 'crew'] # crew 에서 감독정보를 가지고 추출

# 감독 정보를 추출
def get_director(x):
    for i in x:
        if i['job'] == 'Director': # job의 값이 Director 이면 이름값을 추출
            return i['name']
    return np.nan

df2['director'] = df2['crew'].apply(get_director) # get_director 함수로 감독이름을 추출
df2['director']

df2[df2['director'].isnull()] # 감독 정보가 비어있는 것 추출

df2.loc[0, 'cast'] # 주연배우

df2.loc[0, 'genres'] # 장르

df2.loc[0, 'keywords']

# 처음 3개의 데이터 중에서 name 에 해당하는 value 만 추출하는 함수
def get_list(x):
    if isinstance(x, list):  # x가 list 이면
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3] # 처음 3개 3번 인덱스 미만 까지
        return names
    return [] # 리스트가 없으면 빈값으로

features = ['cast', 'keywords', 'genres'] # 함수로 키워드 추출
for feature in features:
    df2[feature] = df2[feature].apply(get_list)

df2[['title', 'cast', 'director', 'keywords', 'genres']].head(3) # 결과

# 모든 텍스트를 소문자로 바꾸고 빈칸은 없애는 함수 만들기
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(' ', '')) for i in x] # replace(' ', '') : 빈칸 없애기
    else:
        if isinstance(x, str): # 문자열인 경우
            return str.lower(x.replace(' ', ''))
        else:
            return ''

features = ['cast', 'keywords', 'director', 'genres'] # 함수 적용. 타일틀은 적용 안함.
for feature in features:
    df2[feature] = df2[feature].apply(clean_data)

df2[['title', 'cast', 'director', 'keywords', 'genres']].head(3) # 결과

def create_soup(x):  # 콤마 대신 빈칸으로 바꾸기
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres']) # cultureclash future spacewar 이런식으로 만들어 주는 작업
df2['soup'] = df2.apply(create_soup, axis=1) # soup 컬럼에 저장
df2['soup']

from sklearn.feature_extraction.text import CountVectorizer # CountVectorizer : 각 단어가 문서에서 몇 번 출현했는지를 기반으로 문서를 벡터화

count = CountVectorizer(stop_words='english') # 영어에 사용하는 관사등 삭제
count_matrix = count.fit_transform(df2['soup'])
count_matrix

from sklearn.metrics.pairwise import cosine_similarity # 진짜 코사인 유사도로 추천 영화 구하기
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
cosine_sim2 # 가장 큰 값이 유사한 영화(추천영화)

indices['Avatar'] # 몇번째 인덱스인지?

df2 = df2.reset_index() # 인덱스 초기화 혹시 데이터가 섞일수 있으므로
indices = pd.Series(df2.index, index=df2['title']) # 새롭게 인덱스 설정 하지 않아도 됨.
indices

get_recommendations('The Dark Knight Rises', cosine_sim2) # cosine_sim2 유사도를 이용한 추천영화

get_recommendations('Up', cosine_sim2)

get_recommendations('The Martian', cosine_sim2)

indices['The Martian'] # The Martian의 영화정보를 알기 위해 인덱스 정보 검색

df2.loc[270] # 마션 영화 데이터

df2.loc[4] # 존카터 영화 정보

get_recommendations('The Avengers', cosine_sim2)

"""여기서 만든 모델을 파일로 만들어 웹사이트로 꾸미기"""

import pickle # Pickle이란 - 텍스트 상태의 데이터가 아닌 파이썬 객체 자체를 바이너리 파일로 저장하는 것을 의미합니다. 속도 빠름.

df2.head(3)

movies = df2[['id', 'title']].copy() # 'id', 'title' 두가지 컬럼만 저장.
movies.head(5)

pickle.dump(movies, open('movies.pickle', 'wb'))  # 파일명은 movies.pickle, 쓰기전용으로 wb

pickle.dump(cosine_sim2, open('cosine_sim.pickle', 'wb')) # 파일명은 osine_sim.pickle, 쓰기전용으로 wb

"""이다음 부터는 VS코드로 할 것. 파이썬 웹사이트 만들기[링크 텍스트](https://docs.google.com/document/d/1C1J44I1No3IeUH5e9OlxZ0dWP_XY_ZoUg_rmqwQu3b8/edit?usp=drive_link)"""

