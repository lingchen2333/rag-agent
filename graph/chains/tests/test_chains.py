from pprint import pprint

from dotenv import load_dotenv

from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import GradeHallucinations, hallucination_grader_chain
from graph.chains.retriever_grader import GradeDocuments, retrieval_grader_chain
from ingestion import retriever

load_dotenv()

def test_retrieval_grader_answer_yes():
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content

    res: GradeDocuments = retrieval_grader_chain.invoke(
        {"question": question, "document": doc_text}
    )

    assert res.binary_score == "yes"

def test_retrieval_grader_answer_no():
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content

    res: GradeDocuments = retrieval_grader_chain.invoke(
        {"question": "how to make pizza", "document": doc_text}
    )

    assert res.binary_score == "no"

def test_generation_chain():
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"question": question, "context": docs})
    pprint(generation)

def test_hallucination_grader_answer_yes():
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"question": question, "context": docs})

    res: GradeHallucinations = hallucination_grader_chain.invoke({
        "documents":docs, "generation":generation}
    )

    assert res.binary_score

def test_hallucination_grader_answer_no():
    question = "agent memory"
    docs = retriever.invoke(question)


    res: GradeHallucinations = hallucination_grader_chain.invoke({
        "documents":docs, "generation":"market capitalization is the total market value of a company"}
    )

    assert not res.binary_score