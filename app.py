import os 
import streamlit as st
import langchain
from typing_extensions import Protocol
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
# print(langchain.__version__)


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
    input_variables = ['title', 'wikipedia_research'],
    template='write me a youtube video script based on this title TITLE:{title} while leveraging this wikipedia research: {wikipedia_research}'
)

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='title_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

llm = OpenAI(temperature=.9)
title_chain = LLMChain(llm=llm, prompt=title_template, 
verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, 
verbose=True, output_key='script', memory=script_memory)

# print("script chain .prompt", script_chain.prompt)
wiki = WikipediaAPIWrapper()
if st.button('Generate'):
    if prompt:
        with st.spinner('Generating response...'):
            title = title_chain.run(topic=prompt)
            wiki_research=wiki.run(prompt)
            
            script = script_chain.run(title=title, wikipedia_research=wiki_research)
            st.write(title)
            st.write(wiki_research)
            # st.write(script)
        with st.expander('Title History'):
            st.info(title_memory.buffer)
        with st.expander('Script History'):
            st.info(script_memory.buffer)
        with st.expander('Wikipedia Research'):
            st.info(wiki_research)
    else:
        st.warning('Please enter your prompt')



