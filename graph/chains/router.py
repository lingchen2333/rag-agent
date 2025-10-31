from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(temperature=0)

class RouteQuery(BaseModel):
    """route a user query to the most relevant datasource"""

    datasource: Literal["vectorstore", "websearch"] = Field(
        description="Given a user question, choose to route it to web search or a vectorstore."
    )

structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or web search. \n 
    The vectorstore contains documents related to agents, prompt engineering and adversarial attacks. \n
    Use the vectorstore for questions on these topics. For everything else, use web search"""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: {question} "),
    ]
)

router_chain = route_prompt | structured_llm_router