#-*- coding:utf-8 -*-
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
from langserve import add_routes
from fastapi import FastAPI

from tools import get_functions

class SampleSchema:
    class Input(BaseModel):
        input_query: str
        chat_history: List[Union[HumanMessage, AIMessage, FunctionMessage]]=[]
        context: str
    class Output(BaseModel):
        output: Any


llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, openai_api_key=api_key)
llm_with_functions = llm.bind(get_functions())

template = """Answer the question based only on the following context:
{context}

Question: {input_query}
"""
prompt = ChatPromptTemplate.from_template(template)

agent = (
    {
        'input_query': lambda x: x['input_query'],
        'agent_scratchpad' : lambda x: format_to_openai_function_messages(x["intermediate_steps"]),
        'context': lambda x: x['context'],
        'chat_history': lambda x: x['chat_history']
    }
    | prompt 
    | llm_with_functions()
    | OpenAIFunctionsAgentOutputParser()
)

executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True,
    early_stopping_method='force', max_execution_time=30, 
    max_iterations=10
)

app = FastAPI(title='sample agent', version='0.1', description='hello')
add_routes(
    app,
    executor.with_types(
        input_type = SampleSchema.Input,
        output_type = SampleSchema.Output
    )
)