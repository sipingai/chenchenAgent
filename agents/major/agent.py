import os
import logging
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from .prompt import Prompt
from .memory import Memory
from .tool import Tool
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Agent:
    def __init__(self):
        self.model_name = "gpt-4o-mini"
        logging.info(f"使用模型: {self.model_name}")
        
        self.chat_model = ChatOpenAI(model=self.model_name)
        self.tool_list = [Tool]
        
        self.memory_key = "major_history"
        self.prompt_structure = Prompt(memory_key=self.memory_key).create_prompt()
        self.memory_instance = Memory(memory_key=self.memory_key).set_memory()

        self.agent = create_tool_calling_agent(
            self.chat_model,
            self.tool_list,
            self.prompt_structure,
        )
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tool_list,
            memory=self.memory_instance,
            verbose=True
        )
        logging.info("代理创建成功")

    def run_agent(self, user_input):
        self.prompt_structure = Prompt(memory_key=self.memory_key).create_prompt()
        logging.info(f"运行代理，输入: {user_input}")
        
        print(self.prompt_structure)
        response = self.agent_executor.invoke({
            "input": user_input,
        })
        logging.info("代理运行完成")
        return response
    
    async def run_agent_ws(self, user_input):
        self.prompt_structure = Prompt(memory_key=self.memory_key).create_prompt()
        logging.info(f"运行WebSocket代理，输入: {user_input}")

        async for event in self.agent_executor.astream_events({"input": user_input, "chat_history": self.memory_instance}, version="v2"):
            event_type = event["event"]
            if event_type == "on_chat_model_stream":
                content_chunk = event["data"]["chunk"].content
                if content_chunk:
                    logging.info(f"接收到内容: {content_chunk}")
                    print(content_chunk, end="|")
                    yield content_chunk