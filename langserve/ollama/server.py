#-*- coding:utf-8 -*-
from typing import List, Any, Union
from langchain_ollama import ChatOllama
from langserve import add_routes
from fastapi import FastAPI

from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.tools import tool, ToolException
from langchain_core.pydantic_v1 import BaseModel, Field


class SampleSchema:
    class Input(BaseModel):
        input_query: str
        chat_history: List[Union[HumanMessage, AIMessage, FunctionMessage]]=[]
        context: str
    class Output(BaseModel):
        output: Any


@tool("sample-tool", return_direct=True)
def sample_tool_function(result:str):
    return result

llm = ChatOllama(model='llama3.1:8b', format='json', temperature=0, keep_alive=-1, base_url='http://127.0.0.1:11434')
tools = [sample_tool_function]
llm_with_tools = llm.bind_tools(tools)

template = """Answer the question based only on the following context:
{context}

Question: {input_query}
"""
prompt = ChatPromptTemplate.from_template(template)

agent = (
    {
        "input_query" : lambda x: x["input_query"],
        "agent_scratchpad" : lambda x: format_to_openai_function_messages(x["intermediate_steps"]),
        "context": lambda x: x['context']
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
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
