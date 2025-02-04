#-*-coding:utf-8 -*-
import google.generativeai as genai
import google.ai.generativelanguage as glm

genai.configure(api_key="AIzaSyD0jMkUWWxa6O1qpGjMIz80zOVxM4KoyKU")

prod_database = { 
    "갤럭시 S26": {"재고": 10, "가격": 1_700_000}, 
    "갤럭시 S25": {"재고": 5, "가격": 1_300_000}, 
    "갤럭시 S24": {"재고": 3, "가격": 1_100_000}, 
} 

def is_product_available(product_name: str)-> bool: 
    """특정 제품의 재고가 있는지 확인한다.
    Args:
        product_name: 제품명
    """

    if product_name in prod_database: 
        if prod_database[product_name]["재고"] > 0: 
            return True 
    return False 

def get_product_price(product_name: str)-> int: 
    """제품의 가격을 가져온다.

    Args:
        product_name: 제품명
    """
    if product_name in prod_database: 
        return prod_database[product_name]["가격"] 
    return None 

def place_order(product_name: str, address: str)-> str: 
    """제품 주문결과를 반환한다.
    Args:
        product_name: 제품명
        address: 배송지
    """
    if is_product_available(product_name): 
        prod_database[product_name]["재고"] -= 1 
        return "주문 완료" 
    else: 
        return "재고 부족으로 주문 불가" 

function_repoistory = {     
    "is_product_available": is_product_available, 
    "get_product_price": get_product_price, 
    "place_order": place_order 
} 

def correct_response(response): 
    part = response.candidates[0].content.parts[0] 
    if part.function_call: 
        for k, v in part.function_call.args.items(): 
            byte_v = bytes(v, "utf-8").decode("unicode_escape") 
            corrected_v = bytes(byte_v, "latin1").decode("utf-8") 
            part.function_call.args.update({k:  corrected_v}) 

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash", 
    tools=function_repoistory.values()
)

chat_session = model.start_chat(history=[])
queries = ["갤럭시 S25 판매 중인가요?", "가격은 어떻게 되나요?", "경기 고양시 일산동구 중앙로1275번길 로 배송부탁드립니다"]             

for query in queries: 
    print(f"\n사용자: {query}")     
    response = chat_session.send_message(query) 
    correct_response(response) 
    part = response.candidates[0].content.parts[0] 

    if part.function_call: 
        function_call =  part.function_call 
        function_name = function_call.name 
        function_args = {k: v for k, v in function_call.args.items()} 
        function_result = function_repoistory[function_name](**function_args) 
        
        part = glm.Part(
            function_response=glm.FunctionResponse(
                name=function_name, 
                response={ 
                    "content": function_result, 
                }, 
            )
        )
        response = chat_session.send_message(part)
        print(f"모델: {response.candidates[0].content.parts[0].text}")