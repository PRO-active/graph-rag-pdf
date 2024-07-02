#まだ未完成
import os
import streamlit as st
from neo4j import GraphDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain.document_loaders import TextLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

# Load and process documents
def load_documents(file_path):
    raw_documents = TextLoader(file_path).load()
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=125)
    return text_splitter.split_documents(raw_documents)

# Create graph documents
def create_graph_documents(documents, llm):
    llm_transformer = LLMGraphTransformer(llm=llm)
    return llm_transformer.convert_to_graph_documents(documents)

# Initialize Neo4j graph and add documents
def initialize_graph(graph_documents, uri, user, password):
    graph = Neo4jGraph(uri=uri, user=user, password=password)
    graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)
    return graph

# Define Streamlit app
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
        # Use existing graph documents
        graph = Neo4jGraph(uri=NEO4J_URI, user=NEO4J_USERNAME, password=NEO4J_PASSWORD)
        st.success("Using existing graph documents.")

    # Display graph
    cypher_query = st.text_input("Enter Cypher query", "MATCH (s)-[r:MENTIONS]->(t) RETURN s,r,t LIMIT 50")
    if st.button("Show Graph"):
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        session = driver.session()
        result = session.run(cypher_query).graph()
        session.close()
        # Use an appropriate library to visualize the graph in Streamlit
        st.graphviz_chart(result)

    # Handle question answering
    question = st.text_input("Ask a question about the document")
    if st.button("Get Answer"):
        # Generate full text query
        def generate_full_text_query(input: str) -> str:
            full_text_query = ""
            words = [el for el in remove_lucene_chars(input).split() if el]
            for word in words[:-1]:
                full_text_query += f" {word}~2 AND"
            full_text_query += f" {words[-1]}~2"
            return full_text_query.strip()

        # Extract entities from text
        class Entities(BaseModel):
            names: List[str] = Field(..., description="Entities in the text")

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract organization and person entities from the text."),
            ("human", "Extract entities from: {question}"),
        ])
        entity_chain = prompt | llm.with_structured_output(Entities)

        def structured_retriever(question: str) -> str:
            entities = entity_chain.invoke({"question": question})
            result = ""
            for entity in entities.names:
                response = graph.query(
                    """CALL db.index.fulltext.queryNodes('entity', $query, {limit:20})
                    YIELD node,score
                    CALL {
                      WITH node
                      MATCH (node)-[r:MENTIONS]->(neighbor)
                      RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                      UNION ALL
                      WITH node
                      MATCH (node)<-[r:MENTIONS]-(neighbor)
                      RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
                    }
                    RETURN output LIMIT 1000
                    """,
                    {"query": generate_full_text_query(entity)},
                )
                result += "\n".join([el['output'] for el in response])
            return result

        # Retrieve data
        structured_data = structured_retriever(question)
        vector_index = Neo4jVector.from_existing_graph(
            OpenAIEmbeddings(),
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
        final_data = f"""Structured data:
        {structured_data}
        Unstructured data:
        {"#Document ". join(unstructured_data)}
        """
        st.write(final_data)

if __name__ == "__main__":
    main()

