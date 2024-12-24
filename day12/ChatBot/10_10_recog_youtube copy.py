# !pip install pytubefix

from pytubefix import YouTube
import os
import google.generativeai as genai

genai.configure(api_key="AIzaSyD0jMkUWWxa6O1qpGjMIz80zOVxM4KoyKU")
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

url = "https://www.youtube.com/watch?v=7qQTyBW4uhI"

def download_youtube(url):
    yt = YouTube(url) # YouTube 객체 생성    
    stream = yt.streams.get_highest_resolution() # 가장 높은 해상도의 스트림 선택
    # 현재 디렉터리에 동영상 다운로드
    file_path = stream.download(output_path="./videos")
    print("Download complete!")
    return file_path
     
import time
import IPython.display  # 동영상 플레이어 출력을 위해 추가

def recog_video(prompt, url, model):
  global file_path, uploaded_file # 이게있어야 delete_file 함수에서 변수를 사용가능
  file_path = download_youtube(url)
  uploaded_file = genai.upload_file(path=file_path)
  contents = [prompt, uploaded_file]
  time.sleep(5)  
  responses = model.generate_content(contents, stream=True, request_options={"timeout": 60*2})

  IPython.display.display(IPython.display.Video(file_path, width=800 ,embed=True))
  for response in responses:
      print(response.text.strip(), end="")  

def delete_file(file_path, uploaded_file):
    if os.path.exists(file_path):
      os.remove(file_path)
      uploaded_file.delete()
      
prompt = """
- 주인공과 영상의 배경을 소설처럼 디테일하게 묘사하세요.
- 주인공이 어느 도시에 있는지 말하고, 왜 그렇게 생각하는지 설명하세요.
- 만일 이 영상이 생성형 AI가 만들었다면, 어떤 부분이 가장 놀랍나요?
"""

recog_video(prompt, url, model)
delete_file(file_path, uploaded_file)

