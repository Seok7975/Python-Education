import google.generativeai as genai

genai.configure(api_key="AIzaSyD0jMkUWWxa6O1qpGjMIz80zOVxM4KoyKU")

# system_instruction = "당신은 유치원 선생님입니다. 사용자는 유치원생입니다. \
#                 쉽고 친절하게 이야기하되 3문장 이내로 짧게 얘기하세요."
# model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=system_instruction)
# chat_session = model.start_chat(history=[])  # ChatSession 객체 반환
# user_queries = ["인공지능이 뭐에요?", "그럼 스스로 생각도 해요?"]

# def correct_response(response): # 한글이 유니코드로 변형되어서 들어가므로 추가
#     part = response.candidates[0].content.parts[0] 
#     if part.function_call: 
#         for k, v in part.function_call.args.items(): 
#             byte_v = bytes(v, "utf-8").decode("unicode_escape") 
#             corrected_v = bytes(byte_v, "latin1").decode("utf-8") 
#             part.function_call.args.update({k:  corrected_v}) 

# for user_query in user_queries:
#     print(f"[사용자]: {user_query}")
#     response = chat_session.send_message(user_query)
#     correct_response(response)
#     print(f"[모델]: {response.text}")
    
import json
system_instruction='JSON schema로 주제별로 답하되 3개를 넘기지 말 것:{{"주제": <주제>,\
                    "답변":<두 문장 이내>}}'
model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=system_instruction, 
                              generation_config={"response_mime_type": "application/json"})
chat_session = model.start_chat(history=[])  # ChatSession 객체 반환
user_queries = ["인공지능의 특징이 뭐에요?", "어떤 것들을 조심해야 하죠?"]

for user_query in user_queries:
    print(f'[사용자]: {user_query}')
    response = chat_session.send_message(user_query)
    answer_dict = json.loads(response.text)
    print(answer_dict)
