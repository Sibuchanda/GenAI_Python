from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

chat_history = [
    SystemMessage('You are a helpful dietitian assistant')
]

while True:
    user_input = input('You : ')
    if(user_input=='exit'):
            break
    chat_history.append(HumanMessage(content=user_input))
    try:
         res = model.invoke(chat_history)
         # chat_history.append(AIMessage(content=res))
         chat_history.append(res)
         print('AI : ',res.content[0]['text'])
    except Exception as e:
          print(f"Error : {e}")




