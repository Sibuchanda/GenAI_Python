from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough

load_dotenv()
parser = StrOutputParser()
model = ChatOpenAI(model="gpt-4o-mini",temperature=0)

prompt1 = PromptTemplate.from_template("tell me a joke about {topic}")

prompt2 = PromptTemplate.from_template("write a 2-line poem about {topic}")

joke_chain = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'poem' : RunnableSequence(prompt2,model,parser)
})

final_chain = RunnableSequence(joke_chain,parallel_chain)
res = final_chain.invoke({'topic':'AI'})

print(res['joke'] + "\n")
print(res['poem'])


