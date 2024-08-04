#まだ未完成
import os
import streamlit as st
from neo4j_utils import initialize_graph, connect_to_neo4j, run_cypher_query
from document_processing import load_documents, create_graph_documents
from rag import handle_question_answering
from langchain.chat_models import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components

def main():
    st.title("PDF to Knowledge Graph with Graph RAG")

    # Initialize session state
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ""
    if 'neo4j_uri' not in st.session_state:
        st.session_state.neo4j_uri = "bolt://localhost:7687"
    if 'neo4j_username' not in st.session_state:
        st.session_state.neo4j_username = "neo4j"
    if 'neo4j_password' not in st.session_state:
        st.session_state.neo4j_password = ""
    if 'neo4j_connected' not in st.session_state:
        st.session_state.neo4j_connected = False
    if 'graph' not in st.session_state:
        st.session_state.graph = None

    # OpenAI API key input
    st.session_state.openai_api_key = st.text_input("Enter OpenAI API Key", value=st.session_state.openai_api_key, type="password")

    # Neo4j connection parameters input
    st.session_state.neo4j_uri = st.text_input("Enter Neo4j URI", value=st.session_state.neo4j_uri)
    st.session_state.neo4j_username = st.text_input("Enter Neo4j Username", value=st.session_state.neo4j_username)
    st.session_state.neo4j_password = st.text_input("Enter Neo4j Password", value=st.session_state.neo4j_password, type="password")

    if st.button("Connect to Neo4j"):
        try:
            # Attempt to connect to Neo4j only when the button is clicked
            graph = connect_to_neo4j(st.session_state.neo4j_uri, st.session_state.neo4j_username, st.session_state.neo4j_password)
            st.session_state.neo4j_connected = True
            st.session_state.graph = graph
            st.success("Connected to Neo4j")
        except Exception as e:
            st.session_state.neo4j_connected = False
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
                    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125", openai_api_key=st.session_state.openai_api_key)
                    llm_transformer = LLMGraphTransformer(llm=llm)  # LLMGraphTransformerを使用
                    graph_documents = create_graph_documents(documents, llm_transformer)
                    
                    # Initialize graph
                    graph = initialize_graph(graph_documents, st.session_state.neo4j_uri, st.session_state.neo4j_username, st.session_state.neo4j_password)
                    st.success("Graph initialized and documents added.")
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")

        else:
            st.success("Using existing graph documents.")

        # Display graph
        cypher_query = st.text_input("Enter Cypher query", "MATCH (s)-[r:MENTIONS]->(t) RETURN s,r,t LIMIT 50")
        if st.button("Show Graph"):
            try:
                driver = GraphDatabase.driver(st.session_state.neo4j_uri, auth=(st.session_state.neo4j_username, st.session_state.neo4j_password))
                session = driver.session()
                result = run_cypher_query(session, cypher_query)
                session.close()
                
                net = Network(height="750px", width="100%", notebook=True)
                for node in result.nodes:
                    net.add_node(node.id, label=node["name"] if "name" in node else node.id)
                for relationship in result.relationships:
                    net.add_edge(relationship.start_node.id, relationship.end_node.id, label=relationship.type)
                
                net.show("graph.html")
                
                # Load the HTML file and display it in Streamlit
                with open("graph.html", "r") as f:
                    html_content = f.read()
                components.html(html_content, height=750)
            except Exception as e:
                st.error(f"Error displaying graph: {e}")

        # Handle question answering
        question = st.text_input("Ask a question about the document")
        if st.button("Get Answer"):
            handle_question_answering(question, st.session_state.graph)

if __name__ == "__main__":
    main()
