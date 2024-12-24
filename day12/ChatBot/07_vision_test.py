# !pip install pillow
import google.generativeai as genai
from PIL import Image

genai.configure(api_key="AIzaSyD0jMkUWWxa6O1qpGjMIz80zOVxM4KoyKU")
image_data = Image.open("./images/monalisa.png") # 모나리자 그림
# model = genai.GenerativeModel('gemini-pro') # 이미지 안됨.
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content(["이 그림에 대해 한 문장으로 설명하세요.", image_data])
print(response.text)

# print(response._result) #response: GenerateContentResponse

# print(f"건수: {len(response.candidates)}")
# print("="*50)
# for candidate in response.candidates:
#     print(candidate)

print(f"finish_reason: {response.candidates[0].finish_reason.name}, {response.candidates[0].finish_reason}")