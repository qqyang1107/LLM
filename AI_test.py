from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.callbacks.base import BaseCallbackHandler

# ✅ 自定義 Callback：即時列印 token
class StreamHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end='', flush=True)

# ✅ 模型名稱（確保你名字正確）
model_name = "a-m-team/AM-Thinking-v1:Q4_K_M"

# ✅ 加入 streaming 回調
stream_handler = StreamHandler()
llm = Ollama(model=model_name, callbacks=[stream_handler], verbose=True)

# ✅ 加入記憶
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

# ✅ 模擬互動
while True:
    user_input = input("\n\nHuman: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        break
    print("AI:", end=' ', flush=True)
    conversation.predict(input=user_input)