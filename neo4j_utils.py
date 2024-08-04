from neo4j import GraphDatabase

def initialize_graph(graph_documents, uri, username, password):
    driver = GraphDatabase.driver(uri, auth=(username, password))
    session = driver.session()
    for doc in graph_documents:
        session.write_transaction(lambda tx: tx.run("CREATE (n:Document {content: $content})", content=doc))
    session.close()

def connect_to_neo4j(uri, username, password):
    driver = GraphDatabase.driver(uri, auth=(username, password))
    session = driver.session()
    session.close()
    return driver

def run_cypher_query(session, query):
    return session.run(query).graph()
