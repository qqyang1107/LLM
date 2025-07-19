from langchain_ollama import ChatOllama  # Changed from OllamaLLM
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent
from langchain.callbacks.base import BaseCallbackHandler
from duckduckgo_search import DDGS

class StreamHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end='', flush=True)

def search_duckduckgo(query, max_results=3):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(f"{r['title']}: {r['href']}\n{r['body']}")
    return "\n\n".join(results) if results else "No results found."

# Define LLM
model_name = "qwen3:14b"
stream_handler = StreamHandler()
llm = ChatOllama(model=model_name, callbacks=[stream_handler])  # Changed to ChatOllama

# Define tools
tools = [
    Tool(
        name="DuckDuckGo Search",
        func=search_duckduckgo,
        description="用來即時搜尋網路資訊，適合查詢最新消息、事實、定義等"
    )
]

# Define memory
checkpointer = MemorySaver()

# Create LangGraph ReAct Agent
agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=checkpointer,
    debug=True
)

while True:
    user_input = input("\n\nHuman: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        break
    print("AI:", end=' ', flush=True)
    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": "conversation_1"}}
    )
    print(response["messages"][-1].content)