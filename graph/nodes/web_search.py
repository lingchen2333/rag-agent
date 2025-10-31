from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

load_dotenv()

web_search_tool = TavilySearch(max_results = 3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---web-search---")
    question = state["question"]
    if "documents" in state:
        documents = state["documents"]
    else:
        documents = None

    tavily_result = web_search_tool.invoke({"query": question})
    joined_tavily_result = "\n".join(
       [res["content"] for res in tavily_result["results"]]
    )

    web_result = Document(page_content=joined_tavily_result)
    if documents is not None:
        documents.append(web_result)
    else:
        documents = [web_result]

    return {"question": question, "documents": documents}

if __name__ == "__main__":
    web_search(state ={"question": "agent memory", "documents": None})