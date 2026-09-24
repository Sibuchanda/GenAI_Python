from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
parser = StrOutputParser()

model = ChatGoogleGenerativeAI(model="gemini-3.7-flash")

# Instantiation using from_template (recommended)
prompt = PromptTemplate.from_template("What is the capital of : {topic}")
# prompt.format(topic="India")

chain = prompt | model | parser
res = chain.invoke({'topic':'India'})
print(res)

# Printing chain graph
chain.get_graph().print_ascii()

