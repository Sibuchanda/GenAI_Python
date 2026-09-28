from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter



load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini",temperature=0)


text = """ One of the most important things I didn't understand about the world when I was a child is the degree to which the returns for performance are superlinear.

Teachers and coaches implicitly told us the returns were linear. "You get out," I heard a thousand times, "what you put in." They meant well, but this is rarely true. If your product is only half as good as your competitor's, you don't get half as many customers. You get no customers, and you go out of business.

It's obviously true that the returns for performance are superlinear in business. Some think this is a flaw of capitalism, and that if we changed the rules it would stop being true. But superlinear returns for performance are a feature of the world, not an artifact of rules we've invented. We see the same pattern in fame, power, military victories, knowledge, and even benefit to humanity. In all of these, the rich get richer. [1]

You can't understand the world without understanding the concept of superlinear returns. And if you're ambitious you definitely should, because this will be the wave you surf on.

It may seem as if there are a lot of different situations with superlinear returns, but as far as I can tell they reduce to two fundamental causes: exponential growth and thresholds. """


splitter = RecursiveCharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=0,
    # separator=''
)

# Note here we used --> split_text because the above text is a string.  But when we want to split text from a pdf then first we have to load the pdf using document loader and as it returns object, that is why we have to use --> split_documents. Basically the input in which we want to perform splliting based on the input type use proper function.
res = splitter.split_text(text)
# print(res)

for text in res:
    print(text)





# Step-by-Step Breakdown
# Default Separators List: RecursiveCharacterTextSplitter automatically uses this list of separators in order:["\n\n", "\n", " ", ""]

# Step 1: Try Paragraphs ("\n\n")It tries to split by double newlines (\n\n). But the paragraphs are hundreds of characters long—far larger than your 10-character limit. So it moves to the next separator.

# Step 2: Try Lines ("\n")It tries single lines (\n), but the lines are still much larger than 10 characters. So it moves to the next separator.

# Step 3: Try Words/Spaces (" ")It tries to split by spaces (" ") to group words together. It fits as many words as possible without exceeding 10 characters:"One of" = 6 characters $\rightarrow$ Fits!"the most" = 8 characters $\rightarrow$ Fits!"important" = 9 characters $\rightarrow$ Fits!"things I" = 8 characters $\rightarrow$ Fits!

# Step 4: Fallback to Single Characters ("") for Long WordsWhat happens when a single word is longer than 10 characters?For example, "understand" is 10 characters, and with surrounding text or spacing, it can't fit into a space-split chunk properly.Word like "performance" is 11 characters (larger than chunk_size=10).Since no space can break an 11-character word, the splitter falls back to its last separator: character-by-character ("").That's why you see long words get cut mid-word: