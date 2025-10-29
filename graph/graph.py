from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

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

workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GENERATE, generate)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEB_SEARCH, web_search)

workflow.set_entry_point(RETRIEVE)
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
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")