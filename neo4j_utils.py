from langchain_community.graphs import Neo4jGraph

def initialize_graph(graph_documents, uri, user, password):
    graph = Neo4jGraph(uri=uri, user=user, password=password)
    graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)
    return graph

def connect_to_neo4j(uri, user, password):
    return Neo4jGraph(uri=uri, user=user, password=password)
