# ollama pull llama3.1:8b
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)
llm = ChatOllama(model="llama3.1:8b", format='json', temperature=0, keep_alive=-1, base_url='http://127.0.0.1:11434')
chain = prompt | llm

ai_message = chain.invoke(
    {
        "input_language": "English",
        "output_language": "German",
        "input": "I love programming."
    }
)
print (ai_message)

