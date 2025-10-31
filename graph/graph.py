from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from graph.chains.answer_grader import answer_grader_chain
from graph.chains.hallucination_grader import hallucination_grader_chain
from graph.chains.router import router_chain
from graph.consts import WEB_SEARCH, GENERATE, RETRIEVE, GRADE_DOCUMENTS
from graph.nodes import generate, retrieve, grade_documents, web_search
from graph.state import GraphState

load_dotenv()

def decide_to_generate(state: GraphState):
    print("---accessing graded documents---")
    if state["web_search"]:
        print("---decision: go to web search---")
        return WEB_SEARCH
    print("---decision: generate---")
    return GENERATE

def grade_generation_grounded_in_documents_and_questions(state: GraphState) -> str:
    print("---checking answer is not hallucinated---")
    documents = state["documents"]
    generation = state["generation"]

    is_grounded = hallucination_grader_chain.invoke(
        {
            "documents": documents,
            "generation": generation,
        }
    )

    if not is_grounded.binary_score:
        print("---decision: generation is not grounded in documents---")
        return "hallucinate"

    print("---decision: generation is grounded in documents---")
    print("---grade generation against question---")

    question = state["question"]
    answer_the_question = answer_grader_chain.invoke(
        {
            "question": question,
            "generation": generation,
        }
    )

    if answer_the_question.binary_score:
        print("---decision: generation addresses the question ---")
        return "useful"
    print("---decision: generation does not the question ---")
    return "not useful"

def route_question(state: GraphState) -> str:
    print("---routing question---")
    question = state["question"]
    res = router_chain.invoke({"question": question})
    if res.datasource == "vectorstore":
        print("---route to vectorstore---")
        return RETRIEVE
    else:
        print("---route to websearch---")
        return WEB_SEARCH


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GENERATE, generate)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEB_SEARCH, web_search)


workflow.set_conditional_entry_point(
    route_question,
    {
        RETRIEVE:RETRIEVE,
        WEB_SEARCH:WEB_SEARCH,
    }
)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        GENERATE:GENERATE,
        WEB_SEARCH:WEB_SEARCH,
    }
)

workflow.add_edge(WEB_SEARCH, GENERATE)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_questions,
    {
        "hallucinate": WEB_SEARCH,
        "not useful": GENERATE,
        "useful":END
    }
)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")