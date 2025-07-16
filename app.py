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

# --- Language code mapping for GoogleTranslator ---
LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Gujarati": "gu",
    "Tamil": "ta"
}

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

# --- LangGraph Nodes ---

# 1. Translation Node (input -> English)
def translate_node(state):
    if state["lang"].lower() != "english":
        translated = GoogleTranslator(source='auto', target='en').translate(state["question"])
    else:
        translated = state["question"]
    state["translated"] = translated
    return state

# 2. Intent Node (detect topic dynamically using Gemini)
def intent_node(state):
    prompt = f"What school subject does the following question belong to?\n\nQuestion: \"{state['translated']}\""
    try:
        response = gemini.generate_content(prompt).text.lower()
        if "math" in response:
            state["topic"] = "Math"
        elif "science" in response:
            state["topic"] = "Science"
        elif "ai" in response or "artificial intelligence" in response:
            state["topic"] = "Artificial Intelligence"
        elif "history" in response:
            state["topic"] = "History"
        elif "geography" in response:
            state["topic"] = "Geography"
        else:
            state["topic"] = response.strip().capitalize()
    except:
        state["topic"] = "General Knowledge"
    state["grade"] = "Grade 6"
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
    topic = state.get("topic", "this topic")
    state["feedback"] = f"🎉 Great job! Want to learn more about {topic}?"
    return state

# 7. Translate Output Back to User's Language
def translate_back_node(state):
    target_lang = LANGUAGE_CODES.get(state["lang"], "en")
    if target_lang != "en":
        try:
            translated_answer = GoogleTranslator(source='auto', target=target_lang).translate(state["answer"])
            translated_feedback = GoogleTranslator(source='auto', target=target_lang).translate(state["feedback"])
            state["answer"] = translated_answer
            state["feedback"] = translated_feedback
        except Exception as e:
            state["answer"] += f"\n\n(⚠️ Translation failed: {str(e)})"
    return state

# --- LangGraph Workflow Setup ---
graph = StateGraph(TutorState)

graph.add_node("translate", RunnableLambda(translate_node))
graph.add_node("intent", RunnableLambda(intent_node))
graph.add_node("retrieve", RunnableLambda(retrieve_node))
graph.add_node("generate", RunnableLambda(generate_answer_node))
graph.add_node("progress", RunnableLambda(progress_node))
graph.add_node("feedback", RunnableLambda(feedback_node))
graph.add_node("translate_back", RunnableLambda(translate_back_node))

graph.set_entry_point("translate")
graph.add_edge("translate", "intent")
graph.add_edge("intent", "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "progress")
graph.add_edge("progress", "feedback")
graph.add_edge("feedback", "translate_back")

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
