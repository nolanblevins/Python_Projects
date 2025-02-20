# Bring in deps
import os 
from apikey import apikey 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('Grant Proposal Writer')
prompt = st.text_input('Plug in your organization info here') 

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='write me a grant proposal title about {topic}'
)
# Abstract Template
abstract_template = PromptTemplate(
    input_variables = ['title', 'organization_research', 'grant_info'], 
    template='write me a grant proposal abstract based on this title TITLE: {title} while leveraging this organization research:{organization_research} and leveraging this info about the grant {grant_info}'
)
# Org Background Template
org_bg_template = PromptTemplate(
    input_variables = ['organization_info'], 
    template='write me a organization background section for a grant proposal based on this {organization_info}'
)
# Project Purpose Template
project_purpose_template = PromptTemplate(
    input_variables = ['organization_info', 'grant_info'], 
    template='write me a project purpose section for a grant proposal based on this {organization_info} while leveraging this grant information {grant_info} '
)
# Budget Use Template
budget_use_template = PromptTemplate(
    input_variables = ['organization_info', 'grant_budget_info'], 
    template='write me a budget use section for a grant proposal based on this {organization_info} while leveraging this grant budget info {grant_budget_info}'
)
# Goals Template
goals_template = PromptTemplate(
    input_variables= ['organization_goals', 'grant_use'],
    template='write me a goals section for a grant proposal based on these organization goals {organization_goals} while leveraging this info about the grant {grant_use}'
)

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
abstract_memory = ConversationBufferMemory(input_key='title''organization_research''grant_info', memory_key='chat_history')
project_purpose_memory = ConversationBufferMemory(input_key='organization_info''grant_info', memory_key='chat_history')
budget_use_memory = ConversationBufferMemory(input_key='organization_info''grant_budget_info', memory_key='chat_history')
goals_memory = ConversationBufferMemory(input_key='organization_goals''grant_use', memory_key='chat_history')

# Llms
llm = OpenAI(temperature=0.9) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
abstract_chain = LLMChain(llm=llm, prompt=abstract_template, verbose=True, output_key='script', memory=abstract_memory)
org_bg_chain = LLMChain(llm=llm, prompt=org_bg_template, verbose=True, output_key='script')
project_purpose_chain = LLMChain(llm=llm, prompt=project_purpose_template, verbose=True, output_key='script', memory=title_memory)
budget_use_chain = LLMChain(llm=llm, prompt=budget_use_template, verbose=True, output_key='script', memory=title_memory)
goals_chain = LLMChain(llm=llm, prompt=goals_template, verbose=True, output_key='script', memory=title_memory)

# Show stuff to the screen if there's a prompt
if prompt: 
    title = title_chain.run(prompt)
    abstract = abstract_chain.run(title=title, organization_research=organization_research)
    org_bg = org_bg_chain.run(organization_info=organization_info)
    project_purpose = project_purpose_chain.run(organization_info=organization_info, grant_info=grant_info)
    budget_use = budget_use_chain.run(organization_info=organization_info, grant_budget_info=grant_budget_info)
    goals = goals_chain.run(organization_goals=organization_goals, grant_use=grant_use)

    st.write(title) 
    st.write(abstract)
    st.write(org_bg)
    st.write(project_purpose)
    st.write(budget_use)
    st.write(goals)

    with st.expander('Title History'): 
        st.info(title_memory.buffer)

    with st.expander('Abstract History'): 
        st.info(abstract_memory.buffer)
        
    with st.expander('Project Purpose History'): 
        st.info(project_purpose.buffer)
        
    with st.expander('Budget Use History'): 
        st.info(budget_use.buffer)
        
    with st.expander('Goals History'): 
        st.info(goals.buffer)