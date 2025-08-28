"""
必要套件安裝指南：
請在執行前確保安裝以下套件，可用以下指令一鍵安裝：

pip install langchain langgraph langchain_ollama duckduckgo_search requests beautifulsoup4
"""

from langchain_ollama import ChatOllama  # 使用 Ollama 模型
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent
from langchain.callbacks.base import BaseCallbackHandler
from duckduckgo_search import DDGS
import math
import requests
from bs4 import BeautifulSoup
import os
import json

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

# 新增：建立資料夾工具
def create_folder(args: dict) -> str:
    path = args.get("path")
    try:
        os.makedirs(path, exist_ok=True)
        return f"資料夾已建立或已存在：{path}"
    except Exception as e:
        return f"Error: {str(e)}"

# 新增：寫入檔案工具
def write_text_file(args: dict) -> str:
    filename = args.get("filename")
    content = args.get("content")
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        return f"已寫入檔案：{filename}"
    except Exception as e:
        return f"Error: {str(e)}"

# 新增：附加寫入檔案工具
def append_text_file(args: dict) -> str:
    filename = args.get("filename")
    content = args.get("content")
    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(content)
        return f"已附加內容至：{filename}"
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
        name="File Reader",
        func=read_file_content,
        description="輸入檔案名稱即可讀取本地文字檔案內容"
    ),
    Tool(
        name="Web Page Reader",
        func=fetch_webpage,
        description="給定 URL 擷取網頁純文字內容（限 1500 字）"
    ),
    Tool(
        name="Create Folder",
        func=create_folder,
        description="輸入 {'path': 資料夾路徑} 可建立新資料夾"
    ),
    Tool(
        name="Write File",
        func=write_text_file,
        description="輸入 {'filename': 檔名, 'content': 內容} 將文字寫入檔案（會覆蓋）"
    ),
    Tool(
        name="Append File",
        func=append_text_file,
        description="輸入 {'filename': 檔名, 'content': 要附加的內容} 將文字加到檔案尾端"
    )
]

# 建立記憶體保存器與 Agent
checkpointer = MemorySaver()
agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=checkpointer,
    debug=False
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
