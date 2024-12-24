import streamlit as st

# if "counter" not in st.session_state:
#     st.session_state.counter = 0

# st.session_state.counter += 1

# st.header(f"This page has run {st.session_state.counter} times.")
# st.button("Run it again")

# 추가입력 4.3.1
# prompt = st.chat_input("메시지를 입력하세요.")

# prompt = st.chat_input("메시지를 입력하세요.")
# if prompt:
#     with st.chat_message("user"):
#         st.write(prompt)

prompt = st.chat_input("메시지를 입력하세요.")
if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("ai", avatar="🤖"):
        st.write("이것은 인공지능 응답입니다.")