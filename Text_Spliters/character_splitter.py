from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter



load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini",temperature=0)


text = """ One of the most important things I didn't understand about the world when I was a child is the degree to which the returns for performance are superlinear.Teachers and coaches implicitly told us the returns were linear. "You get out," I heard a thousand times, "what you put in." They meant well, but this is rarely true. 

If your product is only half as good as your competitor's, you don't get half as many customers. You get no customers, and you go out of business.It's obviously true that the returns for performance are superlinear in business. 

Some think this is a flaw of capitalism, and that if we changed the rules it would stop being true. But superlinear returns for performance are a feature of the world, not an artifact of rules we've invented. We see the same pattern in fame, power, military victories, knowledge, and even benefit to humanity. 

In all of these, the rich get richer. [1]You can't understand the world without understanding the concept of superlinear returns. And if you're ambitious you definitely should, because this will be the wave you surf on.It may seem as if there are a lot of different situations with superlinear returns, but as far as I can tell they reduce to two fundamental causes: exponential growth and thresholds. """


splitter = CharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=0,
    separator=''
)

#Note here we used --> split_text  because the above text is a string.  But when we want to split text from a pdf then first we have to load the pdf   using document loader and as it returns object, that is why we have to use --> split_documents. Basically the input in which we want to perform splliting based on the input type use proper function.
res = splitter.split_text(text)
# print(res)

for text in res:
    print(text)
