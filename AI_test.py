"""
必要套件安裝指南：
請在執行前確保安裝以下套件，可用以下指令一鍵安裝：

pip install langchain langgraph langchain_ollama duckduckgo_search requests beautifulsoup4
"""

from langchain_ollama import ChatOllama  # 使用 Ollama 模型（如 Qwen3）
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent
from langchain.callbacks.base import BaseCallbackHandler
from duckduckgo_search import DDGS
import math
import requests
from bs4 import BeautifulSoup

# 串流回應處理器（即時輸出 LLM token）
class StreamHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end='', flush=True)

# DuckDuckGo 網路搜尋工具
def search_duckduckgo(query, max_results=3):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(f"{r['title']}: {r['href']}\n{r['body']}")
    return "\n\n".join(results) if results else "No results found."

# 計算機工具（支援 math 函式）
def calculator_tool(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": None}, math.__dict__)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# 本地檔案閱讀工具
def read_file_content(filename: str) -> str:
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

# 擷取網頁純文字內容
def fetch_webpage(url: str) -> str:
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator="\n")
        return text[:1500] + ("..." if len(text) > 1500 else "")
    except Exception as e:
        return f"Error: {str(e)}"

# 初始化 LLM
model_name = "qwen3:14b"
stream_handler = StreamHandler()
llm = ChatOllama(model=model_name, callbacks=[stream_handler])

# 定義所有工具
tools = [
    Tool(
        name="DuckDuckGo Search",
        func=search_duckduckgo,
        description="用來即時搜尋網路資訊，適合查詢最新消息、事實、定義等"
    ),
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="可用來計算數學表達式，如『2+2』、『sqrt(9)』等"
    ),
    Tool(
        name="File Reader",
        func=read_file_content,
        description="輸入檔案名稱即可讀取本地文字檔案內容"
    ),
    Tool(
        name="Web Page Reader",
        func=fetch_webpage,
        description="給定 URL 擷取網頁純文字內容（限 1500 字）"
    )
]

# 建立記憶體保存器與 Agent
checkpointer = MemorySaver()
agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=checkpointer,
    debug=True
)

# 主回圈
if __name__ == "__main__":
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
