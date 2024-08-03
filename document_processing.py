from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer

def load_documents(file_path):
    raw_documents = PyPDFLoader(file_path).load()
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=125)
    documents = text_splitter.split_documents(raw_documents)
    return documents

def create_graph_documents(documents, llm_transformer):
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    return graph_documents

