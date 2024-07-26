from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
import streamlit as st

class Entities(BaseModel):
    names: List[str] = Field(..., description="Entities in the text")

def generate_full_text_query(input: str) -> str:
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

def structured_retriever(question: str, graph) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract organization and person entities from the text."),
        ("human", "Extract entities from: {question}"),
    ])
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
    entity_chain = prompt | llm.with_structured_output(Entities)

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

def handle_question_answering(question, graph):
    structured_data = structured_retriever(question, graph)
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
    {"#Document ".join(unstructured_data)}
    """
    st.write(final_data)
