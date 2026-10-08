from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults

load_dotenv()

search_tool = DuckDuckGoSearchResults()
query = input("Ask anything?\n")
res = search_tool.invoke(query)

print(res+"\n")

print(f"Name : {search_tool.name}\n\n")
print(f"Description : {search_tool.description}\n\n")
print(f"Args : {search_tool.args}\n\n")