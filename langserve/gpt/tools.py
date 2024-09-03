import os
import json 
import requests
from typing import Any, List, Union
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain.tools import StructuredTool
from langchain_core.tools import ToolException
from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function

# tool, function
class SampleSchema(BaseModel):
    data : str

def tool_function(data: str):
    return data

sample_tool = StructuredTool.from_function(
    func = tool_function,
    name = 'sample tool',
    description = 'Hello sample tool',
    args_schema  = SampleSchema,
    return_direct = True
)

def get_tools():
    return [sample_tool]

def get_functions():
    tools = get_tools()
    return [format_tool_to_openai_function(t) for t in tools]

