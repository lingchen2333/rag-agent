from typing import Any, Dict

from graph.chains.retriever_grader import retrieval_grader_chain
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determine whether the retrieved documents are relevant to the question.
    If any document is not relevant, we will set a flag to run web search

    args:
        state (dict): The current graph state
    return:
        state (dict): filtered out irrelevant documents and update web_search state
    """

    print("---checking document relevance---")
    question = state["question"]
    documents = state["documents"]

    filtered_documents = []
    web_search = False

    for document in documents:
        score = retrieval_grader_chain.invoke(
            {"question": question, "document": document.page_content}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---grade: document relevant---")
            filtered_documents.append(document)
        else:
            print("---grade: document not relevant---")
            web_search = True

    return {"documents": filtered_documents, "question": question, "web_search": web_search}
