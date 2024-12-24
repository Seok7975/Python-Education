import google.generativeai as genai

genai.configure(api_key="AIzaSyD0jMkUWWxa6O1qpGjMIz80zOVxM4KoyKU")
generation_config = genai.GenerationConfig(stop_sequences=[". ","! "])
model = genai.GenerativeModel('gemini-1.5-flash', generation_config=generation_config)
response = model.generate_content("인공지능에 대해 설명하세요.")

print(response.text)

tokens = model.count_tokens("Learn about language model tokenization.")
print(tokens)