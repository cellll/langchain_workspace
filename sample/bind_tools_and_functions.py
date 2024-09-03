import uuid
from typing import Dict, List, TypedDict, Optional

from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
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
tools = [sample_tool]
functions = [format_tool_to_openai_function(t) for t in tools]

# binding
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, openai_api_key=api_key)
llm_with_tools = llm.bind_tools(tools)
llm_with_functions = llm.bind(functions)

# invoke
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', """작업 내용"""),
        ('user', '{input_query}')
    ]
)
chain = prompt | llm_with_functions
ai_message = chain.invoke({'input_query': '하이'})
print (ai_message)



