from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from neo4j import GraphDatabase
from py2neo import Graph as Py2NeoGraph
import graphviz


class Entities(BaseModel):
    names: List[str] = Field(..., description="Entities in the text")

def visualize_graph(graph):
    dot = graphviz.Digraph()

    for node in graph.nodes:
        dot.node(str(node.identity), str(node["name"]))  # 各ノードを表示

    for rel in graph.relationships:
        dot.edge(str(rel.start_node.identity), str(rel.end_node.identity), label=rel.type)  # 関連を表示

    return dot

def show_graph(graph_uri, graph_username, graph_password):
    try:
        graph = Py2NeoGraph(graph_uri, auth=(graph_username, graph_password))

        query = "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 10"
        result = graph.run(query).graph()

        dot = visualize_graph(result)
        return dot.source
    except Exception as e:
        return f"Error displaying graph: {str(e)}"

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


