from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState):
    print("---generating---")
    question = state["question"]
    docs = state["documents"]

    generation = generation_chain.invoke({"question": question, "context": docs})
    return {"documents": docs, "generation": generation, "question": question}