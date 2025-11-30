import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
model= os.environ.get("LLM_MODEL")
system_prompt = os.environ.get("SYSTEM_PROMPT")

st.set_page_config(
    page_title="Chatbot with Memory",
    page_icon="🤖",
    layout="centered")

st.title("🤖 Chatbot with Memory using Groq")


if "history" not in st.session_state:
    st.session_state.history = []

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

llm = ChatGroq(
    model=model,
    api_key=GROQ_API_KEY,
    temperature=0.0
)


user_prompt = st.chat_input("How can I help you today?")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.history.append({"role": "user", "content": user_prompt})

    # Get the response from the LLM
    response = llm.invoke(
        input = [{"role": "system", "content": system_prompt},
                  *st.session_state.history] )

    st.session_state.history.append({"role": "assistant", "content": response.content})
    with st.chat_message("assistant"):
        st.markdown(response.content)