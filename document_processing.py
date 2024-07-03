from langchain.document_loaders import TextLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

def load_documents(file_path):
    raw_documents = TextLoader(file_path).load()
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=125)
    return text_splitter.split_documents(raw_documents)

def create_graph_documents(documents, llm):
    llm_transformer = LLMGraphTransformer(llm=llm)
    return llm_transformer.convert_to_graph_documents(documents)
