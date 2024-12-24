import streamlit as st
st.title("echo-bot")

# prompt = st.chat_input("메시지를 입력하세요.")
# if prompt:
#     with st.chat_message("user"):
#         st.write(prompt)
#     with st.chat_message("ai", avatar="🤖"):
#         st.write("이것은 인공지능 응답입니다.")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for content in st.session_state.chat_history:
    with st.chat_message(content["role"]):
        st.markdown(content['message'])    

if prompt := st.chat_input("메시지를 입력하세요."):
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "message": prompt})

    with st.chat_message("ai"):                
        response = f'{prompt}... {prompt}... {prompt}...'
        st.markdown(response)
        st.session_state.chat_history.append({"role": "ai", "message": response})
