import google.generativeai as genai 

genai.configure(api_key="AIzaSyD0jMkUWWxa6O1qpGjMIz80zOVxM4KoyKU")
 
model = genai.GenerativeModel('gemini-pro')
chat_session = model.start_chat(history=[]) #ChatSession 객체 반환
user_queries = ["인공지능에 대해 한 문장으로 짧게 설명하세요.", 
                "의식이 있는지 한 문장으로 답하세요."]
for user_query in user_queries:
    print(f'[사용자]: {user_query}')   
    response = chat_session.send_message(user_query)
    print(f'[모델]: {response.text}')

# 수정
for idx, content in enumerate(chat_session.history):
    print(f"{content.__class__.__name__}[{idx}]")
    print(content)