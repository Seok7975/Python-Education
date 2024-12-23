import pickle
import streamlit as st
from tmdbv3api import Movie, TMDb
import os  # 상단에 import 추가

movie = Movie()
tmdb = TMDb()
tmdb.api_key = '5df91a300a31985aad4eeb0480fb25d5' #'your api key'
tmdb.language = 'ko-KR' # 이게 없으면 영어만 뜸

def get_recommendations(title):
    # 영화 제목을 통해서 전체 데이터 기준 그 영화의 index 값을 얻기
    idx = movies[movies['title'] == title].index[0]

    # 코사인 유사도 매트릭스 (cosine_sim) 에서 idx 에 해당하는 데이터를 (idx, 유사도) 형태로 얻기
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 코사인 유사도 기준으로 내림차순 정렬
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # 자기 자신을 제외한 10개의 추천 영화를 슬라이싱
    sim_scores = sim_scores[1:11]
    
    # 추천 영화 목록 10개의 인덱스 정보 추출
    movie_indices = [i[0] for i in sim_scores]
    
    # 인덱스 정보를 통해 영화 제목 추출
    images = []
    titles = []
    for i in movie_indices:
        id = movies['id'].iloc[i]
        details = movie.details(id)
        
        image_path = details['poster_path']
        if image_path:
            image_path = 'https://image.tmdb.org/t/p/w500' + image_path
        else:
            image_path = 'no_image.jpg'

        images.append(image_path)
        titles.append(details['title'])

    return images, titles
    
movies = pickle.load(open('movies.pickle', 'rb'))
cosine_sim = pickle.load(open('cosine_sim.pickle', 'rb'))
# script_dir = os.path.dirname(os.path.abspath(__file__))
# st.write("실행 위치:", os.getcwd())
# st.write("스크립트 위치:", script_dir)

# movies_path = os.path.join(script_dir, 'movies.pickle')
# cosine_sim_path = os.path.join(script_dir, 'cosine_sim.pickle')

try:
    movies = pickle.load(open(movies_path, 'rb'))
    cosine_sim = pickle.load(open(cosine_sim_path, 'rb'))
except Exception as e:
    st.error(f"파일 로드 중 오류 발생: {str(e)}")
    st.write("찾고있는 파일 경로:", movies_path)
    st.stop()
    
st.set_page_config(layout='wide')
st.header('Nadoflix')

movie_list = movies['title'].values
title = st.selectbox('Choose a movie you like', movie_list)
if st.button('Recommend'):
    with st.spinner('Please wait...'):
        images, titles = get_recommendations(title)

        idx = 0
        for i in range(0, 2):
            cols = st.columns(5)
            for col in cols:
                col.image(images[idx])
                col.write(titles[idx])
                idx += 1

 
                
'''
Streamlit앱은 반드시 streamlit run 명령어로 실행해야한다.
streamlit run c:\K_Digital_Training2\Dev\Python\MachineLearning\app.py

혹은

cd c:\K_Digital_Training2\Dev\Python\MachineLearning\app.py 한뒤
steamlit run app.py
'''
