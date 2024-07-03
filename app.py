#まだ未完成
import streamlit as st
from neo4j_utils import initialize_graph, connect_to_neo4j
from document_processing import load_documents, create_graph_documents
from rag import handle_question_answering, generate_full_text_query, structured_retriever

def main():
    st.title("PDF to Knowledge Graph with Graph RAG")

    # Neo4j connection parameters input
    NEO4J_URI = st.text_input("Enter Neo4j URI", "bolt://localhost:7687")
    NEO4J_USERNAME = st.text_input("Enter Neo4j Username", "neo4j")
    NEO4J_PASSWORD = st.text_input("Enter Neo4j Password", type="password")

    if st.button("Connect to Neo4j"):
        st.success("Connected to Neo4j")

    # Option to upload a new PDF or use existing graph documents
    option = st.selectbox("Select input method", ["Use existing graph documents", "Upload a new PDF"])

    if option == "Upload a new PDF":
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        if uploaded_file:
            with open("uploaded_file.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Process PDF and create graph documents
            documents = load_documents("uploaded_file.pdf")
            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
            graph_documents = create_graph_documents(documents, llm)
            
            # Initialize graph
            graph = initialize_graph(graph_documents, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
            st.success("Graph initialized and documents added.")

    else:
        try:
            # Use existing graph documents
            graph = connect_to_neo4j(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
            st.success("Using existing graph documents.")
        except Exception as e:
            st.error(f"Error using existing graph documents: {e}")
            return

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
        handle_question_answering(question, graph)

if __name__ == "__main__":
    main()
  