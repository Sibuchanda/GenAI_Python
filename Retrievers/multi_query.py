from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()


model = ChatOpenAI(model="gpt-4o-mini",temperature=0)
embedding = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store=Chroma(
    collection_name="Random",
    embedding_function=embedding,
    persist_directory='./vector_db'
)

# Relevant health & wellness documents
docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

# vector_store.add_documents(docs)

base_retriver = vector_store.as_retriever(
  search_type="similarity",  # IF we do not use similarity types then answer would be repeated
  search_kwargs={"k":4})
retriver = MultiQueryRetriever.from_llm(
    retriever=base_retriver,
    llm=model
)

query="How can i stay healthy?"
res = retriver.invoke(query)

for i, d in enumerate(res):
    print(f"Result : {i+1}")
    print(d.page_content)


