from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough,RunnableBranch

load_dotenv()
parser = StrOutputParser()
model = ChatOpenAI(model="gpt-4o-mini",temperature=0)

MAX_WORD = 100

prompt1 = PromptTemplate.from_template("Write a detailed report on : {topic}")
prompt2 = PromptTemplate.from_template(f"Summerize the following report within {MAX_WORD} words : {{text}}")

report_chain = RunnableSequence(prompt1,model,parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > MAX_WORD, RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_chain,branch_chain)
res = final_chain.invoke({'topic':'Bitcoin'})
print(res)