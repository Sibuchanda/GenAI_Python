from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


load_dotenv()


# 1. Create Documents
doc1 = Document(
    page_content=(
        "A process is a program in execution. "
        "It has its own memory space and system resources. "
        "The operating system manages processes using process states "
        "such as new, ready, running, waiting, and terminated."
    ),
    metadata={
        "topic": "Process",
        "category": "Process Management"
    }
)


doc2 = Document(
    page_content=(
        "A thread is the smallest unit of CPU execution within a process. "
        "Multiple threads of the same process share resources such as memory "
        "and files. Threads improve concurrency and responsiveness."
    ),
    metadata={
        "topic": "Thread",
        "category": "Process Management"
    }
)


doc3 = Document(
    page_content=(
        "Deadlock occurs when a group of processes are permanently waiting "
        "for resources held by each other. The four necessary conditions "
        "are mutual exclusion, hold and wait, no preemption, and circular wait."
    ),
    metadata={
        "topic": "Deadlock",
        "category": "Process Management"
    }
)


doc4 = Document(
    page_content=(
        "Virtual memory allows an operating system to execute programs "
        "that are larger than the available physical RAM. It uses disk space "
        "as an extension of memory and commonly works with paging."
    ),
    metadata={
        "topic": "Virtual Memory",
        "category": "Memory Management"
    }
)


doc5 = Document(
    page_content=(
        "CPU scheduling determines which ready process should receive the CPU. "
        "Common scheduling algorithms include FCFS, SJF, Round Robin, "
        "and Priority Scheduling. The goal is to improve CPU utilization "
        "and reduce waiting and response time."
    ),
    metadata={
        "topic": "CPU Scheduling",
        "category": "CPU Management"
    }
)


docs = [doc1, doc2, doc3, doc4, doc5]

# 2. Create Embedding Model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# 3. Create Chroma Vector Store
vector_store = Chroma(
    collection_name="os_knowledge",
    embedding_function=embeddings, # Here we link openAI embedding model
    persist_directory="./chroma_db" # create a persistant storage locally , although we have to generate the embedding everytime
)


# 4. Add Documents
vector_store.add_documents(docs)


# 5. View Stored Documents
print("\n========== STORED DOCUMENTS ==========")
# data = vector_store.get()    --> IF we not add 'include' attribute then it will not return embeddings
#  
data = vector_store.get(
    include=["documents", "metadatas",'embeddings']
)


for i in range(len(data["ids"])):
    print(f"\nID: {data['ids'][i]}")
    print("Document:", data["documents"][i])
    print("Metadata:", data["metadatas"][i])
    print("Embeddings:", data["embeddings"][i])

