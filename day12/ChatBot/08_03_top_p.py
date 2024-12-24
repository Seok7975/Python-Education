import google.generativeai as genai

genai.configure(api_key="AIzaSyD0jMkUWWxa6O1qpGjMIz80zOVxM4KoyKU")
# model = genai.GenerativeModel('gemini-pro')
# user_message = "겨울에 대한 짧은 시를 20자 이내로 지으세요."

# print("\ntemperature=0:")
# generation_config = genai.GenerationConfig(temperature=0)
# for _ in range(3):
#     response = model.generate_content(user_message , generation_config=generation_config)
#     print(f'{"="*50}\n{response.text}')

# print("\ntemperature=1:")
# generation_config = genai.GenerationConfig(temperature=1)
# for _ in range(3):
#     response = model.generate_content(user_message , generation_config=generation_config)
#     print(f'{"="*50}\n{response.text}')

# print(genai.get_model("models/gemini-1.5-flash"))

model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content("당신은 뛰어난 연극 배우입니다. 화난 대사를 읊어보세요.")  
print(response._result)