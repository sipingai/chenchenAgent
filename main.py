import os
import uvicorn
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from agents.major.agent import Agent as MajorAgent
from agents.admission.agent import Agent as AdmissionAgent
from models.document import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="Major and Admission API",
    description="Major and Admission API Docs with FastAPI",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redocs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    logging.info("根路由被访问")
    return {"message": "Major and Admission API Docs with FastAPI"}

@app.get("/search_major")
def search_major(request: Request):
    logging.info("搜索专业接口被访问")
    query = request.query_params.get("query", "")
    major_agent = MajorAgent()
    response = major_agent.run_agent(query)
    return response['output']

@app.get("/search_admission")
def search_admission(request: Request):
    logging.info("搜索录取接口被访问")
    query = request.query_params.get("query", "")
    admission_agent = AdmissionAgent()
    response = admission_agent.run_agent(query)
    return response['output']

@app.get("/add_documents")
def add_documents():
    logging.info("添加文档接口被访问")
    try:
        document_module = Document(
            directory_path=os.path.abspath("data"),
            persist_directory=os.path.abspath("databases"),
            collection_name="chroma",
        )
        document_module.add_documents()
        logging.info("文档成功添加")
        return {"message": "文档成功添加"}
    except Exception as e:
        logging.error(f"添加文档时出错：{e}")
        return {"error": "文档添加失败"}

if __name__ == "__main__":
    logging.info("启动服务器")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=1)