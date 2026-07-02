import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        google_api_key=api_key,
    )


st.set_page_config(page_title="Chatbot UI", layout="wide")
st.title("Chatbot — Gemini")

if not api_key:
    st.warning("GOOGLE_API_KEY not set. Create a .env file with GOOGLE_API_KEY.")

llm = get_llm()


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "input" not in st.session_state:
        st.session_state.input = ""


def append_user_message(text: str):
    st.session_state.messages.append({"role": "user", "text": text})


def append_ai_message(text: str):
    st.session_state.messages.append({"role": "ai", "text": text})


def clear_chat():
    st.session_state.messages = []


init_session()

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Conversation")

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['text']}")
        else:
            st.markdown(f"**AI:** {msg['text']}")

    with st.form(key="chat_form", clear_on_submit=False):
        user_input = st.text_area("", value=st.session_state.input, height=100)
        send = st.form_submit_button("Send")
        clear = st.form_submit_button("Clear")

    if clear:
        clear_chat()

    if send and user_input.strip():
        append_user_message(user_input)
        st.session_state.input = ""
        if not api_key:
            append_ai_message("Error: GOOGLE_API_KEY not set.")
        else:
            with st.spinner("Generating response..."):
                try:
                    resp = llm.invoke(user_input)
                    text = getattr(resp, "content", str(resp))
                    append_ai_message(text)
                except Exception as e:
                    append_ai_message(f"Error calling LLM: {e}")

with col2:
    st.subheader("Controls")
    if st.button("Get Career Skills (2026)"):
        if not api_key:
            st.error("Missing GOOGLE_API_KEY — cannot run example.")
        else:
            template = "Give me 3 career skills that are in high demand in {year}."
            prompt_template = PromptTemplate.from_template(template)
            parser = StrOutputParser()
            chain = prompt_template | llm | parser
            try:
                response = chain.invoke({"year": "2026"})
                append_user_message("Example: Career skills in 2026")
                append_ai_message(str(response))
            except Exception as e:
                st.error(f"Chain error: {e}")

    st.markdown("---")
    st.caption("This UI calls the model directly from Streamlit. For production, proxy requests through a backend.")
