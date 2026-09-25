from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

load_dotenv()
parser = StrOutputParser()
model = ChatOpenAI(model="gpt-4o-mini",temperature=0)

prompt1 = PromptTemplate.from_template("What is the capital of : {topic}")
prompt2 = PromptTemplate.from_template("Give me the main reason Why {topic} is called the capital of india?")

chain = RunnableSequence(prompt1,model,parser,prompt2,model,parser)
res = chain.invoke({'topic':'India'})
print(res)

