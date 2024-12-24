import streamlit as st
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 설치된 폰트 
# for font in fm.fontManager.ttflist: print(font.name) # 설치된 폰트 확인
plt.rcParams['font.family'] = 'HYGraphic-Medium'


# 인공신경망 마지막 레이어의 로짓값을 가정
vocab_logits = {"나는": 0.01, "내일": 0.03, "오늘": 0.25, "어제": 0.3,
                "산에": 0.4, "학교에": 0.5, "집에": 0.65,
                "오른다": 1.2, "간다": 1.05, "왔다": 0.95}

# 로짓값을 temperature를 적용하여 확률값으로 변경하는 함수
def softmax_with_temperature(values, temperature):
    epsilon = 1e-2    
    temperature = max(temperature, epsilon)  # 온도가 0이 되는 것을 방지

    # 온도에 따른 소프트맥스 계산
    exp_values = np.exp(np.array(values) / temperature)
    sum_exp_values = np.sum(exp_values)
    softmax_probs = exp_values / sum_exp_values

    return softmax_probs.tolist()


# 바 그래프를 그리는 함수
def draw_prob_graph(vocab, probs):
    # 작은 그림 크기
    fig = plt.figure(figsize=(8, 4))

    # 그라데이션 넣기
    colors = sns.color_palette("Reds", n_colors=len(vocab))

    # 어휘와 확률 정렬
    sorted_vocab_prob = sorted(zip(vocab, probs), key=lambda x: x[1])
    sorted_vocab, sorted_probs = zip(*sorted_vocab_prob)

    #  numpy 배열을 list로 변환
    palette_as_list = [colors[vocab.index(word)] for word in sorted_vocab]

    # 'x'와 동일한 값의 'hue'를 사용하고 범례를 False로 설정하기
    sns.barplot(x=sorted_vocab, y=sorted_probs, hue=sorted_vocab, palette=palette_as_list, dodge=False)
    plt.legend([],[], frameon=False)

    st.pyplot(fig)  # Streamlit을 사용하여 그래프 그리기

# Streamlit 슬라이더를 사용하여 temperature 값을 조정
temperature = st.slider("Temperature 값 조정", min_value=0.01, max_value=100.0, value=1.0, step=0.01, key='temp_slider')
# 로짓에서 확률 분포로 변경
vocab = list(vocab_logits.keys())
logits = list(vocab_logits.values())
probs = softmax_with_temperature(logits, temperature=temperature)

# Markdown을 통해 HTML 코드를 Streamlit에 표시
draw_prob_graph(vocab, probs)

# HTML을 사용한 가운데 정렬
centered_text = "<div style='text-align:center'> 로짓에서 확률분포로 변경된 상태</div>"
st.markdown(centered_text, unsafe_allow_html=True)

# 첫번째 추가 4.2.1
# print("start...")
# import streamlit as st
# text = "마지막 레이어의 로짓값을 가정"
# st.header(text, divider='rainbow')
# st.subheader(text)
# st.title(text)
# st.write(text)
# st.text_input(label="Title", placeholder=text)
# st.write("# Bar Chart")
# vocab_logits = {"나는": 0.01,"내일": 0.03,"오늘": 0.25,"어제": 0.3,
#                 "산에": 0.4,"학교에": 0.5,"집에": 0.65,
#                 "오른다": 1.2,"간다": 1.05,"왔다": 0.95}
# st.bar_chart(vocab_logits)
# st.caption(text)
# print("end...")

# 두번째 추가 5.2.2
import streamlit as st
import time
def get_vocab_logits():
    print(f"get_vocab_logits() starting")
    time.sleep(10)
    vocab_logits = {"나는": 0.01,"내일": 0.03,"오늘": 0.25,"어제": 0.3,
                    "산에": 0.4,"학교에": 0.5,"집에": 0.65,
                    "오른다": 1.2,"간다": 1.05,"왔다": 0.95}
    print(f"get_vocab_logits() ending")
    return vocab_logits

text = "마지막 레이어의 로짓값을 가정"
st.header(text, divider='rainbow')
st.subheader(text)
st.title(text)
st.write(text)
st.text_input(label="Title", placeholder=text)
st.write("# Bar Chart")
st.bar_chart(get_vocab_logits())
st.caption(text)
