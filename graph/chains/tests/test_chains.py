from dotenv import load_dotenv

from graph.chains.retriever_grader import GradeDocuments, retrieval_grader
from ingestion import retriever

load_dotenv()

def test_retrieval_grader_answer_yes():
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_text}
    )

    assert res.binary_score == "yes"

def test_retrieval_grader_answer_no():
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make pizza", "document": doc_text}
    )

    assert res.binary_score == "no"
