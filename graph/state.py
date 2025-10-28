from typing import TypedDict, List


class GraphState(TypedDict):
    """
    represent the state of the graph

    attributes:
        question:
        generation: llm generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]