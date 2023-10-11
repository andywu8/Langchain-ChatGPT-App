import os 
import streamlit as st
import langchain
from typing_extensions import Protocol
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
# from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferMemory
print(langchain.__version__)


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

# Memory 
memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')

llm = OpenAI(temperature=.9)
title_chain = LLMChain(llm=llm, prompt=title_template, 
verbose=True, output_key='title', memory=memory)
script_chain = LLMChain(llm=llm, prompt=script_template, 
verbose=True, output_key='script', memory=memory)
sequential_chain = SequentialChain(chains=[title_chain, script_chain],
input_variables=['topic'], output_variables=['title', 'script'], verbose=True)

if st.button('Generate'):
    if prompt:
        with st.spinner('Generating response...'):
            response = sequential_chain({'topic': prompt}, return_only_outputs=True)
            st.write(response['title'])
            st.write(response['script'])
        with st.expander('Message History'):
            st.info(memory.buffer)
    else:
        st.warning('Please enter your prompt')



