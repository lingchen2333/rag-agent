from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from openai import BaseModel
from pydantic.v1 import Field

load_dotenv()

llm = ChatOpenAI(temperature=0)

class GradeAnswer(BaseModel):
    """binary score for if generated response answers the question"""

    binary_score: bool = Field(
        description="answer addresses the question, 'yes' or 'no'",
    )

structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an answer addresses / resolves a question. \n 
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question. """

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation} "),
    ]
)

answer_grader_chain: RunnableSequence = answer_prompt | structured_llm_grader