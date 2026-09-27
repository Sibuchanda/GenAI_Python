from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
# langchain_community  -->  depricated  -->> use --> langchain_classic
# from langchain_classic.document_loaders import TextLoader



load_dotenv()
parser = StrOutputParser()
model = ChatOpenAI(model="gpt-4o-mini",temperature=0)

loader = PyPDFLoader('sample.pdf')
docs = loader.load()
print(docs[0].page_content)