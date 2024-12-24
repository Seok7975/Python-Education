# st_cache_data_plus.py
import streamlit as st
import time

# total=0

# @st.cache_data
# def get_vocab_logits(param=0):
#     print(f"get_vocab_logits({param}) starting")
#     global total
#     time.sleep(10)
#     vocab_logits = {"나는": 0.01,"내일": 0.03,"오늘": 0.25,"어제": 0.3,
#                     "산에": 0.4,"학교에": 0.5,"집에": 0.65,
#                     "오른다": 1.2,"간다": 1.05,"왔다": 0.95}
#     vocab_logits = {word: logit + param + total for word, logit in vocab_logits.items()}
#     total +=1
#     print(f"get_vocab_logits({param}) ending")    
#     return vocab_logits

# text = "마지막 레이어의 로짓값을 가정"
# st.header(text, divider='rainbow')
# st.subheader(text)
# st.title(text)
# st.write(text)

# user_input = st.number_input(label="로짓값에 더해지는 숫자를 입력하세요.", value=0)

# st.write("# Bar Chart")
# st.bar_chart(get_vocab_logits(user_input))
# st.caption(text)

# # 추가된 사항 4.2.3
# class LinearModel: 
#     def __init__(self, filepath="weight.txt"):
#         self.file_handle = open(filepath, "r")

#     def predict(self, input_data):
#         self.file_handle.seek(0)
#         try:
#             weight = float(self.file_handle.read().strip())
#         except ValueError:
#             weight = 0
#         return input_data * weight + 2

#     def close_file(self):
#         if self.file_handle:
#             self.file_handle.close()
#             self.file_handle = None            
# @st.cache_data
# def load_linear_model():
#     print("loading started...")
#     model = LinearModel()
#     return model

# try:
#     model = load_linear_model()
#     st.write("모델이 정상적으로 로드되었습니다.")
# except Exception as e:
#     st.write(f"Error loading model: {e}")

# 시나리오 비교 코드 4.2.3 주석 번갈아 하면서 테스트 해볼 것.
@st.cache_data
# @st.cache_resource
def get_vocab_logits(param=None):
    print(f"get_vocab_logits({param}) starting")
    vocab_logits = param
    vocab_logits = {"나는": 0.01,"내일": 0.03,"오늘": 0.25,"어제": 0.3,
                    "산에": 0.4,"학교에": 0.5,"집에": 0.65,
                    "오른다": 1.2,"간다": 1.05,"왔다": 0.95}
    print(f"get_vocab_logits({param}) ending")    
    return vocab_logits

user_input = st.number_input(label="'나는'에 대한 로짓값을 입력하세요.", value=0.01)

st.write("# Bar Chart")
vocab_logits = get_vocab_logits()  #(1)함수의 결괏값을 복사하여 반환받음.
vocab_logits['나는'] = user_input  #(2)복사한 값을 변경함
vocab_logits = get_vocab_logits()  #(3)함수의 결괏값을 다시 복사하여 반환받음.
st.bar_chart(vocab_logits)