from langchain_community.graphs import Neo4jGraph

def initialize_graph(graph_documents, url, user, password):
    graph = Neo4jGraph(url=url, username=user, password=password)
    graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)
    return graph

def connect_to_neo4j(url, user, password):
    return Neo4jGraph(url=url, username=user, password=password)