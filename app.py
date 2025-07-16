import streamlit as st
import os
import google.generativeai as genai

from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from langchain_community.retrievers import WikipediaRetriever
from deep_translator import GoogleTranslator

# --- Configure Gemini securely from Streamlit secrets ---
API_KEY = st.secrets["gemini"]["api_key"]
genai.configure(api_key=API_KEY)
gemini = genai.GenerativeModel("gemini-2.0-flash")

# -- LangGraph State Structure --
class TutorState(dict):
    question: str
    lang: str
    translated: str
    topic: str
    grade: str
    docs: str
    answer: str
    feedback: str

# -- LangGraph Nodes --

# 1. Translation Node
def translate_node(state):
    if state["lang"].lower() != "english":
        translated = GoogleTranslator(source='auto', target='en').translate(state["question"])
    else:
        translated = state["question"]
    state["translated"] = translated
    return state

# 2. Intent Node (mock topic/grade)
def intent_node(state):
    state["topic"] = "Math"
    state["grade"] = "Grade 5"
    return state

# 3. Wikipedia Retriever
def retrieve_node(state):
    retriever = WikipediaRetriever()
    docs = retriever.get_relevant_documents(state["translated"])
    state["docs"] = docs[0].page_content if docs else "No content found"
    return state

# 4. Answer Generator using Gemini
def generate_answer_node(state):
    prompt = f"""
You are an AI tutor helping a {state['grade']} student understand the topic: {state['topic']}.
Explain this using very simple words and friendly examples.

Context:
{state['docs']}
"""
    response = gemini.generate_content(prompt)
    state["answer"] = response.text
    return state

# 5. Progress Tracker (can log/save later)
def progress_node(state):
    return state

# 6. Feedback Generator
def feedback_node(state):
    state["feedback"] = f"🎉 Great job! Want to learn more about {state['topic']}?"
    return state

# --- LangGraph Workflow Setup ---
graph = StateGraph(TutorState)

graph.add_node("translate", RunnableLambda(translate_node))
graph.add_node("intent", RunnableLambda(intent_node))
graph.add_node("retrieve", RunnableLambda(retrieve_node))
graph.add_node("generate", RunnableLambda(generate_answer_node))
graph.add_node("progress", RunnableLambda(progress_node))
graph.add_node("feedback", RunnableLambda(feedback_node))

graph.set_entry_point("translate")
graph.add_edge("translate", "intent")
graph.add_edge("intent", "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "progress")
graph.add_edge("progress", "feedback")

chain = graph.compile()



# --- Streamlit Frontend ---
st.set_page_config(page_title="EduGraph Tutor", layout="wide")
st.title("🎓 EduGraph – AI Tutor for Underprivileged Learners")

with st.sidebar:
    st.header("Settings")
    lang = st.selectbox("🌐 Choose Language", ["English", "Hindi", "Gujarati", "Tamil"])
    run = st.button("🧠 Get Answer")

question = st.text_input("📥 Ask your Question")

if "answer" not in st.session_state:
    st.session_state.answer = ""
    st.session_state.feedback = ""

if run and question:
    with st.spinner("Thinking..."):
        result = chain.invoke({
            "question": question,
            "lang": lang
        })
        st.session_state.answer = result["answer"]
        st.session_state.feedback = result["feedback"]

if st.session_state.answer:
    st.markdown("### ✅ AI Tutor Answer")
    st.write(st.session_state.answer)

if st.session_state.feedback:
    st.markdown("### 💬 Feedback")
    st.info(st.session_state.feedback)
