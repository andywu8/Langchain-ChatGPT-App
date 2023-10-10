import os 
import streamlit as st
from typing_extensions import Protocol
from langchain.llms import OpenAI
from dotenv import load_dotenv
load_dotenv()

st.title("LangChain GPT App")
prompt = st.text_input("Enter a prompt")


llm = OpenAI(temperature=.9)
if prompt:
    response = llm(prompt)
    st.write(response)



