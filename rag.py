from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class Entities(BaseModel):
    names: List[str] = Field(..., description="Entities in the text")

def handle_question_answering(question, graph):
    entity_chain = ChatPromptTemplate.from_messages(
        [
            ("system", "You are extracting organization and person entities from the text."),
            ("human", "Use the given format to extract information from the following input: {question}")
        ]
    ) | graph.llm.with_structured_output(Entities)
    
    entities = entity_chain.invoke({"question": question}).names
    return entities

