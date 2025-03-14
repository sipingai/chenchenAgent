import os
import logging
from langchain_community.document_loaders import UnstructuredExcelLoader, Docx2txtLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from stores.Chroma import Store as ChromaStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Document:
    def __init__(self, directory_path=None, persist_directory="chroma_db", collection_name="database"):
        self.directory_path = directory_path
        self.split_texts_list = []
        self.persist_directory = persist_directory
        self.collection_name = collection_name

    def add_documents(self):
        if not self.directory_path:
            logging.warning("未指定目录路径。")
            return None

        loaders = {
            "docx": Docx2txtLoader,
            "pdf": PyPDFLoader,
            "xlsx": UnstructuredExcelLoader,
        }

        for filename in os.listdir(self.directory_path):
            file_path = os.path.join(self.directory_path, filename)
            if os.path.isfile(file_path):
                self.process_file(file_path, filename, loaders)

        return self.create_vector_store()

    def process_file(self, file_path, filename, loaders):
        file_extension = filename.split(".")[-1].lower()
        loader_class = loaders.get(file_extension)

        if loader_class:
            try:
                loader = loader_class(file_path)
                document_text = loader.load()
                self.split_document_texts(document_text)
            except Exception as e:
                logging.error(f"加载 {file_extension} 文件 '{filename}' 时出错：{e}")
        else:
            logging.info(f"不支持的文件扩展名：{file_extension}")

    def split_document_texts(self, document_text):
        if document_text:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
            self.split_texts_list.extend(text_splitter.split_documents(document_text))

    def create_vector_store(self):
        store = ChromaStore(self.persist_directory, self.split_texts_list, self.collection_name)
        logging.info("创建向量存储。")
        return store.create_vector_store()