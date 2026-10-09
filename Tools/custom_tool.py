from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()


# Ste 1 : Using normal 'tool' decorator
@tool
def Multiply(a: int, b: int) -> int:
    """Multiple two numbers"""
    return a*b


res = Multiply.invoke({"a" : 10, "b" : 20})
print(res)

print(f"Name : {Multiply.name}\n")
print(f"Description : {Multiply.description}\n")
print(f"Args : {Multiply.args}")



