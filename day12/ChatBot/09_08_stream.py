# stream.py
import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyD0jMkUWWxa6O1qpGjMIz80zOVxM4KoyKU")
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content("인공지능에 대해 자세히 설명해줘.", stream=True)

for no, chunk in enumerate(response, start=1):
  print(f"{no}: {chunk.text}")
  print("="*50)