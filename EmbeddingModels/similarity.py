from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2",output_dimensionality=768)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about Rohit Sharma'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]   # 1 row(1 query) * 5 columns(5 documents)
index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)




'''

scores
  ↓
[0.8604, 0.8510, 0.8691, 0.8571, 0.8437]

  ↓ enumerate()

(0, 0.8604)
(1, 0.8510)
(2, 0.8691)
(3, 0.8571)
(4, 0.8437)

  ↓ list()

[
    (0, 0.8604),
    (1, 0.8510),
    (2, 0.8691),
    (3, 0.8571),
    (4, 0.8437)
]

  ↓ sorted(key=lambda x: x[1])

[
    (4, 0.8437),
    (1, 0.8510),
    (3, 0.8571),
    (0, 0.8604),
    (2, 0.8691)
]

  ↓ [-1]

(2, 0.8691)

  ↓ unpack

index = 2
score = 0.8691

  ↓ documents[index]

Sachin Tendulkar...



'''