""""topic : langchain : message , invoke , structured output ,prompts """

# prompt = ChatPromptTempplate.from_messages([
#     ("system":"your role is to act as {role}. answer in at max {max} lines"),
#     ("human":"claim : {claim}")
# ])

# prompt.invoke({"role" : "fact checker", "n" : 2, "claim" : "capital of india "})

from __future__ import annotations

import json
import os
from typing import List, Literal, Optional
 
from dotenv import load_dotenv
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from pydantic import BaseModel, Field, ValidationError
 
load_dotenv()

def get_live_llm():
    """Your Veritas config, but returns None instead of crashing when no key is set."""
    if "GEMINI_API_KEY" not in os.environ:
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI
 
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",               
        google_api_key=os.environ["GEMINI_API_KEY"],
        temperature=0,                             
        max_retries=2,                           
        timeout=60,                             
    )

#Messages
def demo_message() -> None:
    msg : List[BaseModel]={
        SystemMessage(content = "you r the fact checker"),
        HumanMessage(content = "is taj mahal in india ? ")
    }
    ai = AIMessage(
        content=get_live_llm().invoke(msg);
    )
    print(ai.content)

#this is generate a answer from 
def fake(*replies: str) -> GenericFakeChatModel:
    """A fake chat model that returns the given strings in order. Perfect for tests."""
    return GenericFakeChatModel(messages=iter(replies))

## prompt template 
def demo_prompts() -> None:
    prompt = ChatPromptTemplate.from_messages{[
        ("system", "You are a {role}. Answer in at most {n} sentences."),
        ("human", "Claim: {claim}"),
    ]}
    value = prompt.invoke({"role": "fact checker", "n": 2, "claim": "The moon is cheese"})

    plcejolder = ChatPromptTemplate.from_messages{[
        ("system" : "you are a research agent"),
        MessagesPlaceholder("history"),
        ("human" : "the moon is cheese")
    ]}
    out = plcejolder.invoke(
        {
            "history": [HumanMessage("hi"), AIMessage("hello!")],
            "question": "what did I just say?",
        }
    ).to_messages()


##### LCEL
#### runables : seriel(normal sytax), parallel(), passthrough()
    
    #     parallel = RunnableParallel(
    #     upper=RunnableLambda(lambda x: x.upper()),
    #     length=RunnableLambda(len),
    # )
    # print("parallel        :", parallel.invoke("veritas"))


### structured output 
class relevance(BaseModel) :
    """to just get the outpus yes or no"""

    sub_que : Literal["yes","no"]= Field(description = "'yes' if the chunk is relevant to the sub-question, else 'no'")

def demo_structured() -> None:
    llm = fake(
        "Sure! Here are the questions: when was it built?",              
        json.dumps({"sub_questions": ["When was it built?", "Who built it?"]}),
    )

PLANNER_SYSTEM_PROMPT = (
    "You are the planning agent for Veritas, a fact-verification system. Break the claim into "
    "3 to 5 specific, independently-researchable sub-questions. Each targets ONE verifiable fact. "
    "Do not answer them."
)
GRADER_SYSTEM_PROMPT = (
    "You are a strict relevance grader. Given a sub-question and one evidence chunk, answer 'yes' "
    "only if the chunk contains substantive information that helps answer the sub-question."
)
 
 
def make_planner(structured_llm):
    """structured_llm: any runnable that returns a PlannerOutput (real or fake)."""
 
    def plan(claim: str) -> List[dict]:
        result: PlannerOutput = structured_llm.invoke(
            [("system", PLANNER_SYSTEM_PROMPT), ("human", f"Claim: {claim}")]
        )
        return [
            {"question": q, "evidence": [], "answer": None, "confidence": None}
            for q in result.sub_questions
        ]
 
    return plan
 
 
def make_grader(structured_llm, max_concurrency: int = 5):
    def grade_many(question: str, chunks: List[str]) -> List[bool]:
        inputs = [
            [("system", GRADER_SYSTEM_PROMPT), ("human", f"Sub-question: {question}\n\nChunk:\n{c}")]
            for c in chunks
        ]
        grades: List[RelevanceGrade] = structured_llm.batch(inputs, config={"max_concurrency": max_concurrency})
        return [g.binary_score == "yes" for g in grades]
 
    return grade_many
 
 
def demo_rebuild(llm) -> None:
    banner("S7: Rebuilt planner + grader (testable with fakes)")
 
    # --- offline: fake "structured LLMs" are just runnables returning the right object ---
    fake_planner_llm = RunnableLambda(
        lambda _msgs: PlannerOutput(sub_questions=["When was it built?", "Who built it?", "Why was it built?"])
    )
    plan = make_planner(fake_planner_llm)
    print("fake plan       :", [sq["question"] for sq in plan("Eiffel Tower built 1889")])
 
    fake_grader_llm = RunnableLambda(
        lambda msgs: RelevanceGrade(binary_score="yes" if "1889" in msgs[1][1] else "no")
    )
    grade_many = make_grader(fake_grader_llm)
    print("fake grades     :", grade_many("When was it built?", ["Completed in 1889.", "Paris has many cafes."]))
 
    # --- live: same functions, real model ---
    if llm is not None:
        live_plan = make_planner(llm.with_structured_output(PlannerOutput))
        print("\nLIVE plan       :", [sq["question"] for sq in live_plan("The Great Wall is visible from space")])