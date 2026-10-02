"""
Research -> Human review (interrupt) -> Email, built with LangGraph + Ollama + Streamlit.

Install:  pip install streamlit langgraph langchain-core langchain-ollama ddgs wikipedia
Model:    ollama pull llama3.1        (any Ollama model that supports tool calling)
Run:      streamlit run app.py
"""
import os
import smtplib
import ssl
import uuid
from email.message import EmailMessage
from typing import Annotated, Literal, TypedDict

import streamlit as st
import wikipedia
from ddgs import DDGS
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt


# ----------------------------------------------------------------------------
# 1. TOOLS (what the LLM can call)
# ----------------------------------------------------------------------------
@tool
def web_search(query: str) -> str:
    """Search the web with DuckDuckGo. Use for recent or general information."""
    try:
        results = DDGS().text(query, max_results=5)
    except Exception as e:
        return f"ERROR: web search failed: {e}"
    if not results:
        return "No web results found."
    return "\n\n".join(f"{r['title']}\n{r['body']}\n{r['href']}" for r in results)[:4000]


@tool
def wikipedia_search(query: str) -> str:
    """Look up a topic on Wikipedia and return a short summary. Use for background facts."""
    try:
        titles = wikipedia.search(query, results=3)
        if not titles:
            return "No Wikipedia page found."
        summary = wikipedia.summary(titles[0], sentences=8, auto_suggest=False)
        return f"Wikipedia - {titles[0]}:\n{summary}"[:4000]
    except wikipedia.DisambiguationError as e:
        return f"Ambiguous topic. Try one of: {', '.join(e.options[:5])}"
    except Exception as e:
        return f"ERROR: wikipedia lookup failed: {e}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send a plain-text email to the given address."""
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "465"))
    user, password = os.getenv("SMTP_USER", ""), os.getenv("SMTP_PASSWORD", "")
    if not user or not password:
        return "ERROR: SMTP user/password not set (see sidebar)."
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = user, to, subject
    msg.set_content(body)
    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context()) as s:
                s.login(user, password)
                s.send_message(msg)
        else:  # e.g. 587 -> STARTTLS
            with smtplib.SMTP(host, port) as s:
                s.starttls(context=ssl.create_default_context())
                s.login(user, password)
                s.send_message(msg)
        return f"Email sent to {to}"
    except Exception as e:
        return f"ERROR: could not send email: {e}"


RESEARCH_TOOLS = [web_search, wikipedia_search]

RESEARCH_PROMPT = (
    "You are a research assistant. For the given topic, call wikipedia_search for background "
    "and web_search for current details (at least once each). When you have enough, STOP calling "
    "tools and write a clear plain-text summary of about 150-250 words with the key facts. "
    "If the reviewer gave feedback, address it (research more if needed) and write a new summary."
)


# ----------------------------------------------------------------------------
# 2. STATE
# ----------------------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # reducer appends messages
    topic: str
    recipient: str
    summary: str
    email_status: str


def get_llm(config: RunnableConfig) -> ChatOllama:
    cfg = config.get("configurable", {})
    return ChatOllama(
        model=cfg.get("model", "llama3.1"),
        base_url=cfg.get("base_url", "http://localhost:11434"),
        temperature=0,
    )


# ----------------------------------------------------------------------------
# 3. NODES
# ----------------------------------------------------------------------------
def researcher(state: State, config: RunnableConfig):
    """LLM decides: call a research tool, or (no tool calls) write the final summary."""
    llm = get_llm(config).bind_tools(RESEARCH_TOOLS)
    ai = llm.invoke([SystemMessage(RESEARCH_PROMPT)] + state["messages"])
    update = {"messages": [ai]}
    if not ai.tool_calls:  # no more tools wanted -> this message IS the summary
        update["summary"] = ai.content if isinstance(ai.content, str) else str(ai.content)
    return update


def route_after_research(state: State) -> Literal["research_tools", "human_review"]:
    last = state["messages"][-1]
    return "research_tools" if getattr(last, "tool_calls", None) else "human_review"


def human_review(state: State) -> Command[Literal["mailer", "researcher"]]:
    """Pause the graph and ask the human. Resumed with Command(resume={...})."""
    decision = interrupt(
        {
            "question": "Is this summary fine?",
            "summary": state["summary"],
            "recipient": state["recipient"],
        }
    )  # <- execution stops here; on resume this returns the resume value
    if decision.get("action") == "approve":
        return Command(goto="mailer", update={"summary": decision.get("summary") or state["summary"]})
    feedback = decision.get("feedback") or "Please improve the summary."
    return Command(
        goto="researcher",
        update={"messages": [HumanMessage(f"Reviewer rejected the summary. Feedback: {feedback}")]},
    )


def mailer(state: State, config: RunnableConfig):
    """LLM tool-calls send_email. We pin 'to' and 'body' so the approved text can't drift."""
    llm = get_llm(config).bind_tools([send_email])
    ai = llm.invoke(
        [
            SystemMessage("Call the send_email tool exactly once. Pick a short, clear subject."),
            HumanMessage(
                f"Recipient: {state['recipient']}\nTopic: {state['topic']}\n\nBody:\n{state['summary']}"
            ),
        ]
    )
    call_args = ai.tool_calls[0]["args"] if ai.tool_calls else {}
    args = {
        "to": state["recipient"],  # enforced
        "subject": call_args.get("subject") or f"Research summary: {state['topic']}",
        "body": state["summary"],  # enforced: send exactly what was approved
    }
    return {"email_status": send_email.invoke(args)}


# ----------------------------------------------------------------------------
# 4. GRAPH
# ----------------------------------------------------------------------------
def build_graph():
    g = StateGraph(State)
    g.add_node("researcher", researcher)
    g.add_node("research_tools", ToolNode(RESEARCH_TOOLS))  # runs the tool calls
    g.add_node("human_review", human_review)
    g.add_node("mailer", mailer)

    g.add_edge(START, "researcher")
    g.add_conditional_edges("researcher", route_after_research, ["research_tools", "human_review"])
    g.add_edge("research_tools", "researcher")  # tool results go back to the LLM
    # human_review has no outgoing edges: it routes itself with Command(goto=...)
    g.add_edge("mailer", END)

    return g.compile(checkpointer=InMemorySaver())  # checkpointer REQUIRED for interrupt()


# ----------------------------------------------------------------------------
# 5. STREAMLIT UI (kept at the end)
# ----------------------------------------------------------------------------
@st.cache_resource
def get_graph():
    # cached so the InMemorySaver (and its paused threads) survive Streamlit reruns
    return build_graph()


def describe_update(node: str, update) -> str:
    if not isinstance(update, dict):
        return f"➡️ {node}"
    msgs = update.get("messages") or []
    last = msgs[-1] if msgs else None
    if node == "researcher" and getattr(last, "tool_calls", None):
        return "🔧 Calling: " + ", ".join(f"`{c['name']}({c['args']})`" for c in last.tool_calls)
    if node == "researcher":
        return "📝 Draft summary ready"
    if node == "research_tools":
        return "📥 Tool results: " + ", ".join(getattr(m, "name", "tool") for m in msgs)
    if node == "mailer":
        return f"📧 {update.get('email_status')}"
    return f"➡️ {node}"


def run(graph, payload, config):
    """Stream the graph until it finishes or hits interrupt(), then update UI state."""
    ss = st.session_state
    status = st.status("Working...", expanded=True)
    try:
        for chunk in graph.stream(payload, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    continue
                line = describe_update(node, update)
                ss.log.append(line)
                status.write(line)
        status.update(label="Done", state="complete")
    except Exception as e:
        status.update(label="Failed", state="error")
        st.error(f"{type(e).__name__}: {e}")
        return

    snap = graph.get_state(config)
    pending = next((i.value for t in snap.tasks for i in t.interrupts), None)
    if pending:
        ss.phase, ss.pending, ss.round = "review", pending, ss.round + 1
    else:
        ss.phase, ss.result = "done", snap.values.get("email_status", "")
    st.rerun()


def main():
    st.set_page_config(page_title="Research → Review → Email", page_icon="📬")
    st.title("📬 Research → Review → Email")
    ss = st.session_state
    for k, v in {"phase": "input", "thread_id": str(uuid.uuid4()), "log": [], "pending": None,
                 "round": 0, "result": "", "topic": ""}.items():
        ss.setdefault(k, v)

    with st.sidebar:
        st.header("Settings")
        model = st.text_input("Ollama model", os.getenv("OLLAMA_MODEL", "llama3.1"))
        base_url = st.text_input("Ollama URL", os.getenv("OLLAMA_URL", "http://localhost:11434"))
        st.divider()
        st.caption("SMTP (Gmail needs an App Password)")
        os.environ["SMTP_HOST"] = st.text_input("SMTP host", os.getenv("SMTP_HOST", "smtp.gmail.com"))
        os.environ["SMTP_PORT"] = st.text_input("SMTP port", os.getenv("SMTP_PORT", "465"))
        os.environ["SMTP_USER"] = st.text_input("Sender email", os.getenv("SMTP_USER", ""))
        os.environ["SMTP_PASSWORD"] = st.text_input("App password", os.getenv("SMTP_PASSWORD", ""), type="password")

    graph = get_graph()
    config = {
        "configurable": {"thread_id": ss.thread_id, "model": model, "base_url": base_url},
        "recursion_limit": 30,
    }

    if ss.phase == "input":
        topic = st.text_input("Topic to research")
        to = st.text_input("Send the summary to (email)")
        if st.button("🔎 Research", type="primary", disabled=not (topic and to)):
            ss.thread_id, ss.log, ss.round, ss.topic = str(uuid.uuid4()), [], 0, topic
            config["configurable"]["thread_id"] = ss.thread_id
            run(graph, {"messages": [HumanMessage(f"Research this topic: {topic}")],
                        "topic": topic, "recipient": to}, config)

    elif ss.phase == "review":
        p = ss.pending
        st.subheader(f"Is this summary fine? — {ss.topic}")
        st.caption(f"Will be emailed to: {p['recipient']}")
        edited = st.text_area("Summary (you can edit before approving)", p["summary"],
                              height=300, key=f"summary_{ss.round}")
        feedback = st.text_input("Not fine? Say what to change", key=f"fb_{ss.round}")
        c1, c2 = st.columns(2)
        if c1.button("✅ Fine — send email", type="primary"):
            run(graph, Command(resume={"action": "approve", "summary": edited}), config)
        if c2.button("🔁 Revise", disabled=not feedback):
            run(graph, Command(resume={"action": "reject", "feedback": feedback}), config)

    else:  # done
        (st.success if ss.result.startswith("Email sent") else st.error)(ss.result)
        if st.button("Start a new topic"):
            ss.phase, ss.log, ss.pending = "input", [], None
            st.rerun()

    if ss.log:
        with st.expander("Activity log"):
            for line in ss.log:
                st.write(line)


main()