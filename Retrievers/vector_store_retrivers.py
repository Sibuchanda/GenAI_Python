from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


load_dotenv()

embedding = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = Chroma(
    collection_name="networking",
    embedding_function=embedding,
    persist_directory="./vector_db"
)


documents = [
    Document(
        page_content="An IP address (Internet Protocol) serves as a unique identifier for a device on a network. It enables devices to locate and communicate with each other across the local network or the internet.",
        metadata={"topic": "Networking", "subject": "IP Address"},
    ),
    Document(
        page_content="DNS (Domain Name System) translates human-friendly domain names like google.com into numerical IP addresses. Without DNS, users would have to memorize complex numerical sequences to visit websites.",
        metadata={"topic": "Networking", "subject": "DNS"},
    ),
    Document(
        page_content="TCP (Transmission Control Protocol) is a connection-oriented protocol that guarantees data delivery in order. It uses a three-way handshake process to establish reliable communication between sender and receiver.",
        metadata={"topic": "Networking", "subject": "TCP"},
    ),
    Document(
        page_content="MAC addresses are unique 48-bit physical identifiers burned into a device's Network Interface Card (NIC) at the factory. Unlike IP addresses, which can change, MAC addresses remain static to identify devices at the local data link layer.",
        metadata={"topic": "Networking", "subject": "MAC Address"},
    ),
    Document(
        page_content="A router operates at Layer 3 of the OSI model and forwards data packets between different computer networks. It analyzes packet headers to determine the best path for data to travel toward its destination.",
        metadata={"topic": "Networking", "subject": "Router"},
    ),
]

# vector_store.add_documents(documents)

retriver = vector_store.as_retriever(search_kwargs={"k":2})
query="where router works?"

res = retriver.invoke(query)

for i, doc in enumerate(res):
    print(f"Result : {i}")
    print(doc.page_content)
