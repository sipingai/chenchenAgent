import logging
from langchain.memory import ConversationTokenBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from .prompt import Prompt
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Memory:
    def __init__(self, memory_key="chat_history", model="gpt-4o-mini"):
        self.memory_key = memory_key
        self.memory = []
        self.chat_model = ChatOpenAI(model=model)
        logger.info("Memory 已初始化，使用模型: %s", model)

    def summary_chain(self, stored_messages):
        system_prompt = Prompt().system_prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt + "\n这是一段你和用户的对话记忆，对其进行总结摘要，然后回答用户的问题："),
            ("user", "{input}")
        ])
        chain = prompt | self.chat_model
        summary = chain.invoke({"input": stored_messages})
        return summary
    
    def get_memory(self):
        try:
            chat_message_history = RedisChatMessageHistory(
                url="redis://localhost:6379/0", session_id="major"
            )
            stored_messages = chat_message_history.messages

            if len(stored_messages) > 10:
                concatenated_messages = "".join(
                    f"{type(message).__name__}: {message.content}" for message in stored_messages
                )
                summary = self.summary_chain(concatenated_messages)
                chat_message_history.clear()
                chat_message_history.add_message(summary)
                logger.info("添加总结后: %s", chat_message_history.messages)
                return chat_message_history
            else:
                logger.info("进入下一步")
                return chat_message_history
        except Exception as e:
            logger.error("获取记忆时出错: %s", e)
            return None

    def set_memory(self):
        self.memory = ConversationTokenBufferMemory(
            llm=self.chat_model,
            human_prefix="user",
            ai_prefix="system",
            memory_key=self.memory_key,
            output_key="output",
            return_messages=True,
            max_token_limit=1000,
            chat_memory=self.get_memory(),
        )
        logger.info("记忆已设置，使用键: %s", self.memory_key)
        return self.memory