from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from neo4j import GraphDatabase

class Entities(BaseModel):
    names: List[str] = Field(..., description="Entities in the text")

def extract_entities_from_question(question, llm):
    entity_chain = ChatPromptTemplate.from_messages(
        [
            ("system", "You are extracting organization and person entities from the text."),
            ("human", "Use the given format to extract information from the following input: {question}")
        ]
    ) | llm.with_structured_output(Entities)

    entities = entity_chain.invoke({"question": question}).names
    return entities

def query_graph(entities, graph_uri, graph_username, graph_password):
    driver = GraphDatabase.driver(graph_uri, auth=(graph_username, graph_password))
    session = driver.session()

    query_result = []
    for entity in entities:
        result = session.run(f"MATCH (n) WHERE n.name CONTAINS '{entity}' RETURN n LIMIT 10")
        query_result.append(result.data())

    session.close()
    return query_result

def handle_question_answering(question, graph_uri, graph_username, graph_password, llm):
    entities = extract_entities_from_question(question, llm)
    if entities:
        result = query_graph(entities, graph_uri, graph_username, graph_password)
        return result
    else:
        return "No entities found in the question."


