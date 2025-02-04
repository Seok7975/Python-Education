import google.generativeai as genai

genai.configure(api_key="AIzaSyD0jMkUWWxa6O1qpGjMIz80zOVxM4KoyKU")
model = genai.GenerativeModel('gemini-pro')
user_queries = [
    {'role': 'user', 'parts': ["인공지능에 대해 40자 이내의 문장으로 설명하세요."]},
    {'role': 'user', 'parts': ["의식이 있는지 40자 이내의 문장으로 답하세요."]}
]
history = []

for user_query in user_queries:
    history.append(user_query)
    print(f'[사용자]: {user_query["parts"][0]}')
    response = model.generate_content(history)    
    # 응답의 길이가 40자를 초과하는 경우 재실행
    while len(response.text) > 40:
        print(f"응답 메시지 길이: {len(response.text)}")
        response = model.generate_content(history)

    print(f'[모델]: {response.text}')
    history.append(response.candidates[0].content)