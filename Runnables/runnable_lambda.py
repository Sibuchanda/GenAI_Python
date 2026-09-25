from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnableLambda, RunnablePassthrough

load_dotenv()
parser = StrOutputParser()
model = ChatOpenAI(model="gpt-4o-mini",temperature=0)

def word_count(word):
    return len(word.split())

prompt1 = PromptTemplate.from_template("tell me a joke about {topic}")

joke_chain = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'len' : RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_chain,parallel_chain)
res = final_chain.invoke({'topic':'duck'})

print(res['joke'])
print(res['len'])