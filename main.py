from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vector import retriever
import streamlit as st

st.title("Pizza Restaurant - Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    model_name = st.selectbox(
        "OllamaModel",
        ["llama3.2", "llama3.1", "llama3.0", "mistral"],
        index=0
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)

    
llm = OllamaLLM(model=model_name, temperature=0.7, streaming=True)

template = """
    You are an expert in answering questions about a pizza restaurant.
    
    Check the following relevant reviews:{reviews}
    
    Here is the question: {question}                       
    """
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm | StrOutputParser()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
    
if user_question := st.chat_input("Ask a question about the pizza restaurant..."):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)
        
    reviews = retriever.invoke(user_question)
        
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_reply = ""
        for token in chain.stream({"reviews": reviews, "question": user_question}):
            full_reply += token
            placeholder.markdown(full_reply + " |")
        placeholder.markdown(full_reply)
    
    st.session_state.messages.append({"role": "assistant", "content": full_reply})
