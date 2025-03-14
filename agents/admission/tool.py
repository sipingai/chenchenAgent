import os
import logging
from langchain.tools import tool
from stores.Chroma import Store

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

chroma_store = Store(os.path.abspath("data"), os.path.abspath("databases"), "chroma")

@tool
def Tool(query):
    """
    Tool: A tool for handling admission queries.

    Description: 用于处理录取查询的工具。
    
    :param query: 用户的查询字符串。
    :return: 从向量存储中获取的结果。
    """
    try:
        logger.info("收到查询请求：%s", query)
        result = chroma_store.query_vector_store(query)
        logger.info("查询结果：%s", result)
        return result
    except Exception as e:
        logger.error("查询过程中出现错误：%s", e)
        return "查询时发生错误，请稍后再试。"