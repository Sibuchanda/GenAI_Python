import os
# 1. Set USER_AGENT before importing WebBaseLoader
os.environ["USER_AGENT"] = "MyLangChainApp/1.0"

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
# langchain_community  -->  depricated  -->> use --> langchain_classic
# from langchain_classic.document_loaders import TextLoader



load_dotenv()
parser = StrOutputParser()
model = ChatOpenAI(model="gpt-4o-mini",temperature=0)

url = "https://www.geeksforgeeks.org/operating-systems/operating-systems-interview-questions/"
loader = WebBaseLoader(url)

prompt1 = PromptTemplate.from_template("Find the answer of the question : {question} from the following text content : {text}")

docs = loader.load()
chain = prompt1 | model | parser
res = chain.invoke(
    {'question':'what is Thread','text':{docs[0].page_content}}
)

print(res)