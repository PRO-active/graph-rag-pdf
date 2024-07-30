from langchain.document_loaders import PyPDFLoader
from typing import List

def load_documents(file_path: str) -> List[str]:
    """Load documents from a PDF file."""
    try:
        pdf_loader = PyPDFLoader(file_path)
        return pdf_loader.load()
    except Exception as e:
        raise RuntimeError(f"Error loading {file_path}") from e

def create_graph_documents(documents, llm):
    llm_transformer = LLMGraphTransformer(llm=llm)
    return llm_transformer.convert_to_graph_documents(documents)
