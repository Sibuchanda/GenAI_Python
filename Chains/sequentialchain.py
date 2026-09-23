from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
parser = StrOutputParser()
model = ChatGoogleGenerativeAI(model="gemini-3.7-flash")

prompt1 = PromptTemplate.from_template("Generate a blog article for topic : {topic}")
prompt2 = PromptTemplate.from_template("Generate summery in two lines from the text : {text}")

chain = prompt1 | model | parser | prompt2 | model | parser
res = chain.invoke({'topic': 'blockchain'})
print(res)

chain.get_graph().print_ascii()
