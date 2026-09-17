from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2",output_dimensionality=32)

# res = embedding.embed_query("Kolkata is the capital of west bengal")
res = embedding.embed_documents([
    "Kolkata is the capital of West Bengal",
    "Bhubhaneswar is the capital of Odisa",
    "Patna is the capital of Bihar"

])
print(res)
