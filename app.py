import os 
import streamlit as st
from typing_extensions import Protocol
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from dotenv import load_dotenv
load_dotenv()

st.title("LangChain GPT App")
prompt = st.text_input("Enter a prompt")

# Prompt Templates
title_template = PromptTemplate(
    input_variables = ['topic'],
    template='write me a youtube video title about {topic}'
)
script_template = PromptTemplate(
    input_variables = ['title'],
    template='write me a youtube video script based on this title TITLE:{title}'
)


llm = OpenAI(temperature=.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True)
sequential_chain = SimpleSequentialChain(chains=[title_chain, script_chain], verbose=True)

if prompt:
    response = sequential_chain.run(prompt)
    st.write(response)



