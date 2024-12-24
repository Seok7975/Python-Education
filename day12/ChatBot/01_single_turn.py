import google.generativeai as genai

genai.configure(api_key="AIzaSyD0jMkUWWxa6O1qpGjMIz80zOVxM4KoyKU")
model = genai.GenerativeModel('gemini-pro')
# model = genai.GenerativeModel('gemini-1.5-flash') 
response = model.generate_content("인공지능에 대해 한 문장으로 설명하세요.")

print(response.text)