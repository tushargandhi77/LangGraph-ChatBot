from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests

load_dotenv()



llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash') 


# Create the tools

# Tools

search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num:float,second_num:float,operation: str)->dict:
    """
    perform a basic arithmetic operation on two numbers.
    Supported operations: add, subtract, mul, and div
    """

    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed."}
            result = first_num / second_num
        else:
            return {"error": "Unsupported operation. Please use add, sub, mul, or div."}
        
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
    

@tool
def get_stock_price(symbol: str)->dict:
    """
    Fetch latest stock price for a given symbol (e.g 'AAPL','TSLA')
    using Alpha vantage with API key in the URL
    """

    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=43M0JWCXQK4D3L4N"
    r = requests.get(url)

    return r.json()

# Make tool list
tools = [get_stock_price,calculator,search_tool]

llm_with_tools = llm.bind_tools(tools)

#################################################################################


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


### Nodes

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

######################################

conn = sqlite3.connect(database='chatbot.db',check_same_thread=False)
# Checkpointer
checkpointer = SqliteSaver(conn=conn)


# Graph 
# graph Structure

graph = StateGraph(ChatState)
graph.add_node('chat_node',chat_node)
graph.add_node('tools',tool_node)


graph.add_edge(START,'chat_node')

graph.add_conditional_edges("chat_node",tools_condition)

graph.add_edge('tools','chat_node')

chatbot = graph.compile(checkpointer=checkpointer)


#####################################################

def retrieve_all_threads():
    all_threads = set()

    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)


