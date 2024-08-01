#まだ未完成
import os
import streamlit as st
from neo4j_utils import initialize_graph, connect_to_neo4j
from document_processing import load_documents, create_graph_documents
from rag import handle_question_answering
from langchain.chat_models import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer

def main():
    st.title("PDF to Knowledge Graph with Graph RAG")

    # OpenAI API key input
    OPENAI_API_KEY = st.text_input("Enter OpenAI API Key", type="password")

    # Neo4j connection parameters input
    NEO4J_URI = st.text_input("Enter Neo4j URI", "bolt://localhost:7687")
    NEO4J_USERNAME = st.text_input("Enter Neo4j Username", "neo4j")
    NEO4J_PASSWORD = st.text_input("Enter Neo4j Password", type="password")

    if "neo4j_connected" not in st.session_state:
        st.session_state.neo4j_connected = False

    if st.button("Connect to Neo4j"):
        try:
            # Attempt to connect to Neo4j only when the button is clicked
            graph = connect_to_neo4j(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
            st.session_state.neo4j_connected = True
            st.session_state.graph = graph
            st.success("Connected to Neo4j")
        except Exception as e:
            st.error(f"Error connecting to Neo4j: {e}")

    if st.session_state.neo4j_connected:
        # Option to upload a new PDF or use existing graph documents
        option = st.selectbox("Select input method", ["Use existing graph documents", "Upload a new PDF"])

        if option == "Upload a new PDF":
            uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
            if uploaded_file and st.button("Process PDF"):
                # Save the uploaded file to the local disk
                pdf_path = "uploaded_file.pdf"
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Process PDF and create graph documents
                try:
                    documents = load_documents(pdf_path)  # Use the saved file path
                    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125", openai_api_key=OPENAI_API_KEY)
                    llm_transformer = LLMGraphTransformer(llm=llm)  # LLMGraphTransformerを使用
                    graph_documents = create_graph_documents(documents, llm_transformer)
                    
                    # Initialize graph
                    graph = initialize_graph(graph_documents, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
                    st.success("Graph initialized and documents added.")
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")

        else:
            st.success("Using existing graph documents.")

        # Display graph
        cypher_query = st.text_input("Enter Cypher query", "MATCH (s)-[r:MENTIONS]->(t) RETURN s,r,t LIMIT 50")
        if st.button("Show Graph"):
            try:
                driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
                session = driver.session()
                result = session.run(cypher_query).graph()
                session.close()
                # Use an appropriate library to visualize the graph in Streamlit
                st.graphviz_chart(result)
            except Exception as e:
                st.error(f"Error displaying graph: {e}")

        # Handle question answering
        question = st.text_input("Ask a question about the document")
        if st.button("Get Answer"):
            handle_question_answering(question, st.session_state.graph)

if __name__ == "__main__":
    main()