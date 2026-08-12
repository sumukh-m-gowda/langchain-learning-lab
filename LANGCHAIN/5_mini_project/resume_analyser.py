from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

# ─────────────────────────────────────────────
# PYDANTIC SCHEMA  (Structured Output)
# ─────────────────────────────────────────────
class ResumeAnalysis(BaseModel):
    match_score: int = Field(
        gt=0, lt=101,
        description="A score from 1 to 100 indicating how well the resume matches the job description"
    )
    verdict: Literal["Strong Match", "Moderate Match", "Weak Match"] = Field(
        description="Overall verdict based on the match score"
    )
    strengths: list[str] = Field(
        description="List of strengths in the resume that align well with the job description"
    )
    weaknesses: list[str] = Field(
        description="List of gaps or weaknesses in the resume compared to the job description"
    )
    missing_keywords: list[str] = Field(
        description="Important keywords or skills from the job description that are missing in the resume"
    )
    suggested_improvements: list[str] = Field(
        description="Specific actionable suggestions to improve the resume for this job"
    )
    summary: str = Field(
        description="A 2-3 sentence overall summary of the analysis"
    )


# ─────────────────────────────────────────────
# MODEL + PARSER + CHAIN SETUP
# ─────────────────────────────────────────────
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

parser = PydanticOutputParser(pydantic_object=ResumeAnalysis)

# ChatPromptTemplate with MessagesPlaceholder for follow-up chat history
analysis_prompt = ChatPromptTemplate([
    ("system", """You are an expert HR consultant and resume coach with 15+ years of experience.
Analyse the provided resume against the job description thoroughly and honestly.
Be specific, practical, and constructive in your feedback.

{format_instruction}"""),
    ("human", """JOB DESCRIPTION:
{job_description}

RESUME:
{resume}

Provide a detailed analysis.""")
])

# StrOutputParser chain for the follow-up Q&A chatbot
chat_prompt = ChatPromptTemplate([
    ("system", "You are an expert resume coach. You have already analysed a resume against a job description. Answer follow-up questions using that context."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

analysis_chain = analysis_prompt | model | parser
chat_chain = chat_prompt | model | StrOutputParser()


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="Resume Analyser", page_icon="📄", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0f0f13; color: #e8e8f0; }
    .block-container { padding: 2rem 3rem; max-width: 1200px; }

    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .score-box {
        background: linear-gradient(135deg, #1e1b4b, #1e3a5f);
        border: 1px solid #4338ca44;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .score-number {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .verdict-badge {
        display: inline-block;
        padding: 0.35rem 1.2rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .verdict-strong { background: #14532d; color: #86efac; }
    .verdict-moderate { background: #451a03; color: #fcd34d; }
    .verdict-weak { background: #450a0a; color: #fca5a5; }

    .card {
        background: #16161e;
        border: 1px solid #ffffff0f;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 0.75rem;
    }
    .tag {
        display: inline-block;
        background: #1e1b4b;
        color: #a5b4fc;
        border-radius: 6px;
        padding: 0.25rem 0.7rem;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    .tag-red {
        background: #450a0a;
        color: #fca5a5;
    }
    .list-item {
        padding: 0.4rem 0;
        border-bottom: 1px solid #ffffff08;
        font-size: 0.9rem;
        color: #d1d5db;
    }
    .list-item:last-child { border-bottom: none; }
    .chat-bubble-user {
        background: #1e1b4b;
        border-radius: 12px 12px 4px 12px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        color: #e8e8f0;
        font-size: 0.9rem;
    }
    .chat-bubble-ai {
        background: #16161e;
        border: 1px solid #ffffff0f;
        border-radius: 12px 12px 12px 4px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        color: #d1d5db;
        font-size: 0.9rem;
    }
    .stTextArea textarea {
        background: #16161e !important;
        color: #e8e8f0 !important;
        border: 1px solid #ffffff18 !important;
        border-radius: 10px !important;
        font-size: 0.9rem !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #2563eb);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-family: 'Space Grotesk', sans-serif;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    div[data-testid="stVerticalBlock"] { gap: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ─── Header ───
st.markdown('<p class="hero-title">Resume Analyser</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Paste a job description and your resume — get an honest, structured breakdown in seconds.</p>', unsafe_allow_html=True)

# ─── Session State ───
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

# ─── Input Section ───
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Job Description**")
    jd = st.text_area("", height=280, placeholder="Paste the job description here...", key="jd_input", label_visibility="collapsed")
with col2:
    st.markdown("**Your Resume**")
    resume = st.text_area("", height=280, placeholder="Paste your resume text here...", key="resume_input", label_visibility="collapsed")

analyse_clicked = st.button("Analyse Resume →")

if analyse_clicked:
    if not jd.strip() or not resume.strip():
        st.warning("Please fill in both the job description and your resume.")
    else:
        with st.spinner("Analysing your resume..."):
            try:
                result = analysis_chain.invoke({
                    "job_description": jd,
                    "resume": resume,
                    "format_instruction": parser.get_format_instructions()
                })
                st.session_state.analysis = result
                st.session_state.jd_text = jd
                st.session_state.resume_text = resume
                # seed chat history with context
                st.session_state.chat_history = [
                    SystemMessage(content=f"Resume analysed. JD: {jd[:500]}... Resume: {resume[:500]}..."),
                    AIMessage(content=f"I've analysed the resume. Match score: {result.match_score}/100. Verdict: {result.verdict}. Summary: {result.summary}")
                ]
            except Exception as e:
                st.error(f"Something went wrong: {e}")

# ─── Results Section ───
if st.session_state.analysis:
    r = st.session_state.analysis
    st.markdown("---")
    st.markdown("### Analysis Results")

    # Score + Verdict + Summary
    c1, c2 = st.columns([1, 2])
    with c1:
        badge_class = (
            "verdict-strong" if r.verdict == "Strong Match"
            else "verdict-moderate" if r.verdict == "Moderate Match"
            else "verdict-weak"
        )
        st.markdown(f"""
        <div class="score-box">
            <div style="font-size:0.75rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.1em;">Match Score</div>
            <div class="score-number">{r.match_score}<span style="font-size:1.5rem;color:#4b5563">/100</span></div>
            <span class="verdict-badge {badge_class}">{r.verdict}</span>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card" style="height:100%">
            <div class="card-title">Summary</div>
            <p style="color:#d1d5db;font-size:0.95rem;line-height:1.6">{r.summary}</p>
        </div>
        """, unsafe_allow_html=True)

    # Strengths + Weaknesses
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        items = "".join([f'<div class="list-item">✅ {s}</div>' for s in r.strengths])
        st.markdown(f'<div class="card"><div class="card-title">Strengths</div>{items}</div>', unsafe_allow_html=True)
    with col_b:
        items = "".join([f'<div class="list-item">⚠️ {w}</div>' for w in r.weaknesses])
        st.markdown(f'<div class="card"><div class="card-title">Weaknesses</div>{items}</div>', unsafe_allow_html=True)

    # Missing Keywords + Suggested Improvements
    col_c, col_d = st.columns(2)
    with col_c:
        tags = "".join([f'<span class="tag tag-red">{k}</span>' for k in r.missing_keywords])
        st.markdown(f'<div class="card"><div class="card-title">Missing Keywords</div>{tags}</div>', unsafe_allow_html=True)
    with col_d:
        items = "".join([f'<div class="list-item">💡 {s}</div>' for s in r.suggested_improvements])
        st.markdown(f'<div class="card"><div class="card-title">Suggested Improvements</div>{items}</div>', unsafe_allow_html=True)

    # ─── Follow-up Chat ───
    st.markdown("---")
    st.markdown("### Ask a Follow-up Question")
    st.markdown('<p style="color:#6b7280;font-size:0.85rem">Ask anything about your results — e.g. "How do I fix my biggest weakness?" or "What skills should I add?"</p>', unsafe_allow_html=True)

    # Render chat history (skip the seeded SystemMessage/AIMessage)
    for msg in st.session_state.chat_history[2:]:
        if isinstance(msg, HumanMessage):
            st.markdown(f'<div class="chat-bubble-user">🧑 {msg.content}</div>', unsafe_allow_html=True)
        elif isinstance(msg, AIMessage):
            st.markdown(f'<div class="chat-bubble-ai">🤖 {msg.content}</div>', unsafe_allow_html=True)

    question = st.text_area("Your question", height=80, placeholder="e.g. How can I improve my chances for this role?", label_visibility="collapsed")

    if st.button("Ask →"):
        if question.strip():
            with st.spinner("Thinking..."):
                st.session_state.chat_history.append(HumanMessage(content=question))
                reply = chat_chain.invoke({
                    "chat_history": st.session_state.chat_history,
                    "question": question
                })
                st.session_state.chat_history.append(AIMessage(content=reply))
                st.rerun()