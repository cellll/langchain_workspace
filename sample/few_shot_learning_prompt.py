import uuid
from typing import Dict, List, TypedDict, Optional

from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, openai_api_key=api_key)


# few shot
example_prompt = ChatPromptTemplate.from_messages([("human", "{input_query}"),("ai", "{output}")])
examples = [
    {'input_query': "입력 문장1", 'output' : '결과1'},
    {'input_query': "입력 문장2", 'output' : '결과2'},
    {'input_query': "입력 문장3", 'output' : '결과3'}
]    
few_shot_prompt = FewShotChatMessagePromptTemplate( example_prompt=example_prompt, examples=examples)


# prompt 
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', """작업 내용"""),
        few_shot_prompt,
        ('user', '{input_query}')
    ]
)

# invoke
chain = prompt | llm
ai_message = chain.invoke({'input_query':'하이'})
print (ai_message)
