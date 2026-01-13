from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model="llama-3.1-8b-instant",
    model_provider="groq",
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "{input}")
    ]
)

parser = StrOutputParser()

chain = prompt | model | parser

response = chain.invoke({"input": "What is Machine Learning"})

print(response)
