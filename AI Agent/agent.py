from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
import requests

from langchain_classic.agents import create_react_agent, AgentExecutor
from langsmith import Client

client = Client()

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)

search_tool = DuckDuckGoSearchResults()

@tool
def get_district_name(pincode: int) -> str:
    """ This function fetches the District name of the given pincode """
    url=f"https://api.postalpincode.in/pincode/{pincode}"
    # User-Agent prevents the remote server from abruptly cutting the connection
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        
        # CRITICAL FIX: Extract the text from the response object so the LLM agent can read it
        return res.text 
        
    except Exception as e:
        # Prevents the application terminal from crashing if the API is offline
        return f"Error retrieving data for pincode: {str(e)}"


 # Safely pull the public prompt using the modern SDK
prompt = client.pull_prompt("hwchase17/react", dangerously_pull_public_prompt=True)

# Step 2: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_district_name],
    prompt=prompt
)

# Step 3: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_district_name],
    verbose=True
)

# Step 4: Invoke
response = agent_executor.invoke({"input": "Find the district name from the pincode : 700001  and then find the state name for this fetched district name"})
print(response['output'])
