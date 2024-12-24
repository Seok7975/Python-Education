# !pip install pytubefix

from pytubefix import YouTube

url = "https://www.youtube.com/watch?v=i-E7NiyRDa0"

yt = YouTube(url) # YouTube 객체 생성    
stream = yt.streams.get_highest_resolution() # 가장 높은 해상도의 스트림 선택
file_path = stream.download(output_path="./videos")
print("Download complete!")
     
import google.generativeai as genai

# 여기 있어야 작동됨.
model = genai.GenerativeModel(model_name="gemini-1.5-flash")
genai.configure(api_key="AIzaSyD0jMkUWWxa6O1qpGjMIz80zOVxM4KoyKU")

uploaded_file = genai.upload_file(path=file_path)
print("uploaded_file.uri:", uploaded_file.uri)

import IPython.display  # 동영상 플레이어 출력을 위해 추가

prompt = """
유튜브를 보고 아래에 답하세요.
- 영상에 등장하는 춤을 추는 인물은 몇 명인가요?
- 각각의 인물에 대한 특징을 짧게 기술하세요.
"""
import time # 안되면 삽입
contents = [prompt, uploaded_file]
time.sleep(5)  # 안되면 삽입
responses = model.generate_content(contents, stream=True, request_options={"timeout": 60*4})

IPython.display.display(IPython.display.Video(file_path, width=800 ,embed=True))
for response in responses:
    print(response.text.strip(), end="")
    
import  os

if os.path.exists(file_path):
      os.remove(file_path)

uploaded_file.delete()
