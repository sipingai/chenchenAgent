import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

class Store:
    def __init__(self, persist_directory, split_texts_list, collection_name):
        self.persist_directory = persist_directory
        self.split_texts_list = split_texts_list
        self.collection_name = collection_name

    def create_vector_store(self):
        embeddings = OpenAIEmbeddings()

        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)

        vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings,
            collection_name=self.collection_name
        )

        if not vector_store:
            vector_store.add_documents(self.split_texts_list, collection_name=self.collection_name)
            vector_store.persist()

        return vector_store
    
    def query_vector_store(self, query_text):
        vector_store = self.create_vector_store()
        results = vector_store.search(query_text)
        return results