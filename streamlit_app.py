import streamlit as st
from rag_engine import answer_question

st.set_page_config(page_title="Local RAG Chat", page_icon="🤖")

st.title("🤖 Local PDF Chat")
st.caption("Offline • GPU • PDF-grounded")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask from your PDFs...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = answer_question(prompt)
            st.markdown(answer)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
