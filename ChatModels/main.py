from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.7-flash")

res = model.invoke("What is the capital of india")
print(res.content[0]['text'])
