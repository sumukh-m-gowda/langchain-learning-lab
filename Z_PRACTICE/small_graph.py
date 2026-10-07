"""
Veritas: planner -> researcher -> grader -> verdict, wired in sequence with LangGraph, powered by Gemini.

Install:
    pip install langgraph langchain-google-genai pydantic ddgs

Run:
    export GOOGLE_API_KEY="your-key"
    python veritas.py "The Great Wall is visible from space"
"""
import os
import sys
from typing import List, Literal, Optional, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

MODEL_NAME = "gemini-2.5-flash"

# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------
PLANNER_SYSTEM_PROMPT = (
    "You are the planning agent for Veritas, a fact-verification system. Break the claim into "
    "3 to 5 specific, independently-researchable sub-questions. Each targets ONE verifiable fact. "
    "Do not answer them."
)

GRADER_SYSTEM_PROMPT = (
    "You are a strict relevance grader. Given a sub-question and one evidence chunk, answer 'yes' "
    "only if the chunk contains substantive information that helps answer the sub-question."
)

VERDICT_SYSTEM_PROMPT = (
    "You are the verdict agent for Veritas. You receive a claim and, for each sub-question, a set "
    "of graded-relevant evidence chunks. Answer each sub-question using ONLY the evidence given "
    "(say 'insufficient evidence' if there is none), give a confidence from 0 to 1, then give an "
    "overall verdict on the claim: 'supported', 'refuted', 'mixed', or 'insufficient'."
)


# --------------------------------------------------------------------------------------
# Structured output schemas
# --------------------------------------------------------------------------------------
class PlannerOutput(BaseModel):
    sub_questions: List[str] = Field(description="3 to 5 independently researchable sub-questions")


class RelevanceGrade(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="'yes' if the chunk helps answer the sub-question")


class SubAnswer(BaseModel):
    question: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)


class VerdictOutput(BaseModel):
    sub_answers: List[SubAnswer]
    verdict: Literal["supported", "refuted", "mixed", "insufficient"]
    explanation: str


# --------------------------------------------------------------------------------------
# Graph state
# --------------------------------------------------------------------------------------
class VeritasState(TypedDict, total=False):
    claim: str
    sub_questions: List[dict]  # {"question", "evidence", "answer", "confidence"}
    verdict: Optional[dict]


# --------------------------------------------------------------------------------------
# Node factories
# --------------------------------------------------------------------------------------
def make_planner(structured_llm):
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
        if not chunks:
            return []
        inputs = [
            [("system", GRADER_SYSTEM_PROMPT), ("human", f"Sub-question: {question}\n\nChunk:\n{c}")]
            for c in chunks
        ]
        grades: List[RelevanceGrade] = structured_llm.batch(
            inputs, config={"max_concurrency": max_concurrency}
        )
        return [g.binary_score == "yes" for g in grades]

    return grade_many


def web_search(query: str, max_results: int = 6) -> List[dict]:
    """Returns [{"text": ..., "source": ...}]. Swap this for Tavily, a vector store, etc. if you like."""
    try:
        from ddgs import DDGS
    except ImportError:  # older package name
        from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        hits = ddgs.text(query, max_results=max_results) or []
    return [
        {"text": f"{h.get('title', '')}: {h.get('body', '')}".strip(), "source": h.get("href", "")}
        for h in hits
        if h.get("body")
    ]


# --------------------------------------------------------------------------------------
# LangGraph nodes
# --------------------------------------------------------------------------------------
def planner_node_factory(plan_fn):
    def planner_node(state: VeritasState) -> dict:
        sub_questions = plan_fn(state["claim"])
        print(f"\n[planner] {len(sub_questions)} sub-questions")
        for sq in sub_questions:
            print("  -", sq["question"])
        return {"sub_questions": sub_questions}

    return planner_node


def researcher_node(state: VeritasState) -> dict:
    updated = []
    for sq in state["sub_questions"]:
        chunks = web_search(sq["question"])
        print(f"[researcher] {len(chunks)} chunks for: {sq['question']}")
        updated.append({**sq, "evidence": chunks})
    return {"sub_questions": updated}


def grader_node_factory(grade_many_fn):
    def grader_node(state: VeritasState) -> dict:
        updated = []
        for sq in state["sub_questions"]:
            chunks = sq["evidence"]
            flags = grade_many_fn(sq["question"], [c["text"] for c in chunks])
            kept = [c for c, ok in zip(chunks, flags) if ok]
            print(f"[grader] kept {len(kept)}/{len(chunks)} for: {sq['question']}")
            updated.append({**sq, "evidence": kept})
        return {"sub_questions": updated}

    return grader_node


def verdict_node_factory(verdict_llm):
    def verdict_node(state: VeritasState) -> dict:
        blocks = []
        for i, sq in enumerate(state["sub_questions"], 1):
            ev = "\n".join(f"  [{c['source']}] {c['text']}" for c in sq["evidence"]) or "  (no relevant evidence)"
            blocks.append(f"Sub-question {i}: {sq['question']}\nEvidence:\n{ev}")
        prompt = f"Claim: {state['claim']}\n\n" + "\n\n".join(blocks)

        result: VerdictOutput = verdict_llm.invoke(
            [("system", VERDICT_SYSTEM_PROMPT), ("human", prompt)]
        )

        by_q = {a.question: a for a in result.sub_answers}
        updated = []
        for sq in state["sub_questions"]:
            a = by_q.get(sq["question"])
            updated.append(
                {**sq, "answer": a.answer if a else None, "confidence": a.confidence if a else None}
            )
        return {
            "sub_questions": updated,
            "verdict": {"verdict": result.verdict, "explanation": result.explanation},
        }

    return verdict_node


# --------------------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------------------
def build_graph(llm):
    plan_fn = make_planner(llm.with_structured_output(PlannerOutput))
    grade_fn = make_grader(llm.with_structured_output(RelevanceGrade))
    verdict_llm = llm.with_structured_output(VerdictOutput)

    g = StateGraph(VeritasState)
    g.add_node("planner", planner_node_factory(plan_fn))
    g.add_node("researcher", researcher_node)
    g.add_node("grader", grader_node_factory(grade_fn))
    g.add_node("verdict", verdict_node_factory(verdict_llm))

    g.add_edge(START, "planner")
    g.add_edge("planner", "researcher")
    g.add_edge("researcher", "grader")
    g.add_edge("grader", "verdict")
    g.add_edge("verdict", END)
    return g.compile()


def main() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        sys.exit("Set GOOGLE_API_KEY first.")

    claim = " ".join(sys.argv[1:]) or "The Great Wall is visible from space"
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)

    app = build_graph(llm)
    final = app.invoke({"claim": claim})

    print("\n" + "=" * 70)
    print("CLAIM  :", claim)
    print("VERDICT:", final["verdict"]["verdict"].upper())
    print("WHY    :", final["verdict"]["explanation"])
    print("-" * 70)
    for sq in final["sub_questions"]:
        conf = f"{sq['confidence']:.2f}" if sq["confidence"] is not None else "n/a"
        print(f"Q: {sq['question']}\nA: {sq['answer']}  (confidence {conf})")
        for c in sq["evidence"]:
            print("   source:", c["source"])
        print()


if __name__ == "__main__":
    main()