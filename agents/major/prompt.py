import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Prompt:
    def __init__(self, memory_key="chat_history"):
        self.memory_key = memory_key
        self.system_prompt = """你是一个大学专业查询助手，专门帮助用户了解和选择适合他们的大学专业。
        以下是你的个人设定：
        1. 你对世界各地的大学和专业有广泛的了解，可以提供专业介绍、课程信息和职业前景。
        2. 你能够根据用户的兴趣和背景，推荐合适的专业。
        3. 你总是以清晰、简洁和友好的方式回答问题。
        4. 你可以用简体中文或英语回答问题，视用户的需求而定。
        5. 你从不透露自己是一个人工智能助手，而是以专业顾问自居。
        """
        logging.info("Prompt 初始化完成")

    def create_prompt(self):
        logging.info(f"使用的记忆键: {self.memory_key}")
        
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name=self.memory_key),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ]
        )
        logging.info("提示结构创建成功")
        return prompt_template