from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader  
# langchain_community  -->  depricated  -->> use --> langchain_classic
# from langchain_classic.document_loaders import TextLoader



load_dotenv()
parser = StrOutputParser()
model = ChatOpenAI(model="gpt-4o-mini",temperature=0)


prompt1 = PromptTemplate.from_template("Give me a short summery of the following text : {topic}")

loader = TextLoader('content.txt')
docs = loader.load()

# print(type(docs))
# print(docs[0].page_content)
# print(docs[0].metadata)

chain = prompt1 | model | parser
res = chain.invoke({'topic': docs[0].page_content}) 
print(res)
