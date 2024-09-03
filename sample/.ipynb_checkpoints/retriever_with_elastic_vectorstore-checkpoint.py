import json 
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.documents.base import Document
from langchain.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI

from datetime import timedelta
import warnings
warnings.filterwarnings(action='ignore')
from elasticsearch import Elasticsearch
from langchain_elasticsearch import ElasticsearchStore




# embedding model
model_name = '/son/embedding_models/bge-m3'
model_kwargs = {'device':'cpu'}
encode_kwargs = {"normalize_embeddings": True}
embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# elastic vector store 
client = Elasticsearch(
    'https://127.0.0.1:9200',
    basic_auth=("elastic", 'password'),
    verify_certs=False
)

elastic = ElasticsearchStore(
    es_connection=client,
    index_name='test_vectorstore',
    embedding=embedding
)

# tools, functions
class SearchSchema(BaseModel):
    docs: List[Document]=Field(default=[], description='Documents what you choose for answer the question')
    answer: str=Field(default="", description="Your answer for question")


def search_func(docs: List[Document], answer: str):
    doc_contents = list()
    for doc in docs:
        print (doc.metadata)
        print (doc.page_content)
        doc_content = f"[{doc.metadata['start']}~{doc.metadata['end']}] {doc.page_content}"
        doc_contents.append(doc_content)

    result = {
        'answer' : answer,
        'relevant_docs' : doc_contents
    }
    return result


search_tool = StructuredTool.from_function(
    func=search_func,
    name='search_tool',
    description='Make a answer based on context data.',
    args_schema=SearchSchema,
    return_direct=True
)

tools = [search_tool]
functions = [convert_to_openai_tool(tool)['function'] for tool in tools]

# binding
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, openai_api_key=api_key)
llm_with_functions = llm.bind(functions)

template = """Answer the question based only on the following context:
{context}

Question: {input_query}
"""
prompt = ChatPromptTemplate.from_template(template)
# #### 1
# chain = prompt | llm_with_functions

# # invoke with context
# ai_message = chain.invoke(
#     {
#         'input_query': '하이',
#         'context' : elastic.similarity_search('하이')
#     }
# )
# print (ai_message)

#### 2 
chain = (
    {'context' : lambda x: elastic.similarity_search(x['input_query']),  'input_query' : RunnablePassthrough()}
    | prompt
    | llm_with_functions 
)
ai_message = chain.invoke(
    {
        'input_query': '하이'
    }
)


# embedding query 사용 시 
# elastic.similarity_search_by_vector_with_relevance_scores(query_emb, k=20, filter = [{'term' : {'metadata.type' : 'video'}}])


