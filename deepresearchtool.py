from langchain_core.tools import Tool
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate # Import PromptTemplate
from langchain.chains import LLMChain # Import LLMChain
import json
from langchain.agents import initialize_agent, Tool
from custom_search import safe_search

key = YOUR-API-KEY
llm = ChatOpenAI(api_key = key, temperature = 0, model = "gpt-4")

#def query
example  = """
1) Query Generator:
    - What are the possible causes of severe stomach pain?
    - What are the common symptoms associated with severe stomach pain?
    - What are the potential treatments for severe stomach pain?

2) Websearch Agent:
    - Search for the most common causes of severe stomach pain from trusted government medical sites.
    - Find information on the symptoms that are commonly associated with severe stomach pain.
    - Look for the recommended treatments for severe stomach pain from reputable medical websites.

3) Document Retriever Agent:
    - Retrieve documents that discuss the causes of severe stomach pain.
    - Find documents that detail the symptoms associated with severe stomach pain.
    - Search for documents that outline the potential treatments for severe stomach pain.

4) Summarizer Agent:
    - Summarize the information found on the causes of severe stomach pain.
    - Provide a summary of the symptoms associated with severe stomach pain.
    - Summarize the recommended treatments for severe stomach pain.
"""

def planner_agent(query):
    modified_prompt = PromptTemplate.from_template("""
    You are a deep medical research planner. WHen the user gives you a query your job is to break it down into subquestions. Your job is to plan the deep research based on the {query} of the user.
    You will create subquestions that will be sent to multiple agents for context so they can provide better answers./n
    Make the plan based on the following agents. Give a plan for every agent don't go beyond that.

    1)Query Generator-Reword the query based on the context from an internal database
    2)Websearch Agent - Find the relevant information by scrapping information from government medical sites
    3)Doucment Retriever Agent - Use vector search from internal documents to match the most similar result
    4)Summarizer Agent - Will summarize the informaiton from 2) and 3) and provide personalized solution

    Questions : "{query}"

    Return the sub-questions as a numbered list
    """)
    decomposer_chain = LLMChain(llm = llm, prompt = modified_prompt)
    response = decomposer_chain.run({"query": query})
    print(response)
planner_agent(query = "I have bad stomach pain")


def query_generator(context,query):
  medical_data = match_medical_category(query)
  modified_prompt = PromptTemplate.from_template("""
  You are responsible for regenerating the query based on in-house context. You will use the following information

  1){context}- You will these questions while redoing the query
  2){medical_data} - This takes information from a inhouse database about the user

  You are job as query generator agent is to combine 1) and 2) and redo {query}
  """
  )
  decomposer_chain = LLMChain(llm = llm, prompt = modified_prompt)
  response = decomposer_chain.run({"query", query})

def safe_search(query : str) -> str:
  allowed_domains = [
      "medlineplus.gov"
  ]
  results = 


def websearch_agent(context,query):
  tools = Tool(name )
  
  
  modified_prompt = PromptTemplate.from_template("""
  
    
  
  
  """)
